#include "cm_dataset.h"

#include <errno.h>
#include <limits.h>
#include <math.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define CM_ZIP_LOCAL_SIG   UINT32_C(0x04034b50)
#define CM_ZIP_CENTRAL_SIG UINT32_C(0x02014b50)
#define CM_ZIP_EOCD_SIG    UINT32_C(0x06054b50)
#define CM_NPY_PREFIX_SIZE 10u
#define CM_MEMBER_COUNT 8u

static const char *const cm_member_names[CM_MEMBER_COUNT] = {
    "X.npy", "Y.npy", "g.npy", "seeds.npy",
    "n_sim.npy", "n_pop.npy", "perc_rc.npy", "sb.npy"
};

static void cm_set_error(char *error, size_t error_size, const char *format, ...)
{
    va_list args;
    if (error == NULL || error_size == 0u) {
        return;
    }
    va_start(args, format);
    (void)vsnprintf(error, error_size, format, args);
    va_end(args);
    error[error_size - 1u] = '\0';
}

static int cm_mul_size(size_t a, size_t b, size_t *result)
{
    if (a != 0u && b > SIZE_MAX / a) {
        return 0;
    }
    *result = a * b;
    return 1;
}

static int cm_add_size(size_t a, size_t b, size_t *result)
{
    if (b > SIZE_MAX - a) {
        return 0;
    }
    *result = a + b;
    return 1;
}

static int cm_range_ok(size_t offset, size_t length, size_t total)
{
    return offset <= total && length <= total - offset;
}

static uint16_t cm_get_u16le(const unsigned char *p)
{
    return (uint16_t)((uint16_t)p[0] | ((uint16_t)p[1] << 8));
}

static uint32_t cm_get_u32le(const unsigned char *p)
{
    return (uint32_t)p[0] |
           ((uint32_t)p[1] << 8) |
           ((uint32_t)p[2] << 16) |
           ((uint32_t)p[3] << 24);
}

static uint64_t cm_get_u64le(const unsigned char *p)
{
    return (uint64_t)p[0] |
           ((uint64_t)p[1] << 8) |
           ((uint64_t)p[2] << 16) |
           ((uint64_t)p[3] << 24) |
           ((uint64_t)p[4] << 32) |
           ((uint64_t)p[5] << 40) |
           ((uint64_t)p[6] << 48) |
           ((uint64_t)p[7] << 56);
}

static void cm_put_u16le(unsigned char *p, uint16_t value)
{
    p[0] = (unsigned char)(value & UINT16_C(0xff));
    p[1] = (unsigned char)((value >> 8) & UINT16_C(0xff));
}

static void cm_put_u32le(unsigned char *p, uint32_t value)
{
    p[0] = (unsigned char)(value & UINT32_C(0xff));
    p[1] = (unsigned char)((value >> 8) & UINT32_C(0xff));
    p[2] = (unsigned char)((value >> 16) & UINT32_C(0xff));
    p[3] = (unsigned char)((value >> 24) & UINT32_C(0xff));
}

static void cm_put_u64le(unsigned char *p, uint64_t value)
{
    p[0] = (unsigned char)(value & UINT64_C(0xff));
    p[1] = (unsigned char)((value >> 8) & UINT64_C(0xff));
    p[2] = (unsigned char)((value >> 16) & UINT64_C(0xff));
    p[3] = (unsigned char)((value >> 24) & UINT64_C(0xff));
    p[4] = (unsigned char)((value >> 32) & UINT64_C(0xff));
    p[5] = (unsigned char)((value >> 40) & UINT64_C(0xff));
    p[6] = (unsigned char)((value >> 48) & UINT64_C(0xff));
    p[7] = (unsigned char)((value >> 56) & UINT64_C(0xff));
}

static uint32_t cm_crc32(const unsigned char *data, size_t size)
{
    static uint32_t table[256];
    static int initialized = 0;
    uint32_t crc = UINT32_C(0xffffffff);
    size_t i;

    if (!initialized) {
        uint32_t n;
        for (n = 0u; n < 256u; ++n) {
            uint32_t c = n;
            unsigned k;
            for (k = 0u; k < 8u; ++k) {
                c = (c & 1u) ? (UINT32_C(0xedb88320) ^ (c >> 1)) : (c >> 1);
            }
            table[n] = c;
        }
        initialized = 1;
    }
    for (i = 0u; i < size; ++i) {
        crc = table[(crc ^ data[i]) & UINT32_C(0xff)] ^ (crc >> 8);
    }
    return crc ^ UINT32_C(0xffffffff);
}

static int cm_read_file(const char *path,
                        unsigned char **data_out,
                        size_t *size_out,
                        char *error,
                        size_t error_size)
{
    FILE *file;
    long length;
    unsigned char *data;
    size_t got;

    *data_out = NULL;
    *size_out = 0u;
    file = fopen(path, "rb");
    if (file == NULL) {
        cm_set_error(error, error_size, "cannot open '%s': %s", path, strerror(errno));
        return 0;
    }
    if (fseek(file, 0L, SEEK_END) != 0 || (length = ftell(file)) < 0L ||
        fseek(file, 0L, SEEK_SET) != 0) {
        cm_set_error(error, error_size, "cannot determine size of '%s'", path);
        (void)fclose(file);
        return 0;
    }
#if ULONG_MAX > SIZE_MAX
    if ((unsigned long)length > SIZE_MAX) {
        cm_set_error(error, error_size, "file '%s' is too large for this build", path);
        (void)fclose(file);
        return 0;
    }
#endif
    data = (unsigned char *)malloc((size_t)length == 0u ? 1u : (size_t)length);
    if (data == NULL) {
        cm_set_error(error, error_size, "out of memory reading '%s'", path);
        (void)fclose(file);
        return 0;
    }
    got = fread(data, 1u, (size_t)length, file);
    if (got != (size_t)length) {
        cm_set_error(error, error_size, "short read from '%s'", path);
        free(data);
        (void)fclose(file);
        return 0;
    }
    if (fclose(file) != 0) {
        cm_set_error(error, error_size, "error closing '%s'", path);
        free(data);
        return 0;
    }
    *data_out = data;
    *size_out = (size_t)length;
    return 1;
}

typedef struct CmBitReader {
    const unsigned char *data;
    size_t size;
    size_t byte_pos;
    unsigned bit_pos;
} CmBitReader;

static int cm_bit(CmBitReader *reader, unsigned *value)
{
    if (reader->byte_pos >= reader->size) {
        return 0;
    }
    *value = ((unsigned)reader->data[reader->byte_pos] >> reader->bit_pos) & 1u;
    ++reader->bit_pos;
    if (reader->bit_pos == 8u) {
        reader->bit_pos = 0u;
        ++reader->byte_pos;
    }
    return 1;
}

static int cm_bits(CmBitReader *reader, unsigned count, unsigned *value)
{
    unsigned result = 0u;
    unsigned i;
    for (i = 0u; i < count; ++i) {
        unsigned bit;
        if (!cm_bit(reader, &bit)) {
            return 0;
        }
        result |= bit << i;
    }
    *value = result;
    return 1;
}

static void cm_bit_align_byte(CmBitReader *reader)
{
    if (reader->bit_pos != 0u) {
        reader->bit_pos = 0u;
        ++reader->byte_pos;
    }
}

typedef struct CmHuffman {
    uint16_t counts[16];
    uint16_t first_code[16];
    uint16_t first_symbol[16];
    uint16_t symbols[320];
    int empty;
} CmHuffman;

static int cm_huffman_build(CmHuffman *table,
                            const unsigned char *lengths,
                            size_t symbol_count,
                            char *error,
                            size_t error_size)
{
    uint32_t code = 0u;
    uint32_t symbol_base = 0u;
    int slots = 1;
    size_t symbol;
    unsigned length;
    uint16_t next[16];

    if (symbol_count > sizeof(table->symbols) / sizeof(table->symbols[0])) {
        cm_set_error(error, error_size, "DEFLATE Huffman alphabet is too large");
        return 0;
    }
    memset(table, 0, sizeof(*table));
    for (symbol = 0u; symbol < symbol_count; ++symbol) {
        if (lengths[symbol] > 15u) {
            cm_set_error(error, error_size, "invalid DEFLATE Huffman code length");
            return 0;
        }
        ++table->counts[lengths[symbol]];
    }
    table->empty = table->counts[0] == symbol_count;
    if (table->empty) {
        return 1;
    }
    /* Zero-length symbols are absent; they never participate in canonical codes. */
    table->counts[0] = 0u;
    for (length = 1u; length <= 15u; ++length) {
        slots = (slots << 1) - (int)table->counts[length];
        if (slots < 0) {
            cm_set_error(error, error_size, "oversubscribed DEFLATE Huffman tree");
            return 0;
        }
        code = (code + table->counts[length - 1u]) << 1;
        if (code > UINT16_MAX) {
            cm_set_error(error, error_size, "invalid DEFLATE Huffman code set");
            return 0;
        }
        table->first_code[length] = (uint16_t)code;
        table->first_symbol[length] = (uint16_t)symbol_base;
        symbol_base += table->counts[length];
        next[length] = table->first_symbol[length];
    }
    for (symbol = 0u; symbol < symbol_count; ++symbol) {
        length = lengths[symbol];
        if (length != 0u) {
            table->symbols[next[length]++] = (uint16_t)symbol;
        }
    }
    return 1;
}

static int cm_huffman_decode(CmBitReader *reader,
                             const CmHuffman *table,
                             unsigned *symbol_out)
{
    uint32_t code = 0u;
    unsigned length;
    if (table->empty) {
        return 0;
    }
    for (length = 1u; length <= 15u; ++length) {
        unsigned bit;
        uint32_t first;
        uint32_t count;
        if (!cm_bit(reader, &bit)) {
            return 0;
        }
        code = (code << 1) | bit;
        first = table->first_code[length];
        count = table->counts[length];
        if (code >= first && code - first < count) {
            *symbol_out = table->symbols[table->first_symbol[length] + (code - first)];
            return 1;
        }
    }
    return 0;
}

static int cm_fixed_trees(CmHuffman *literal,
                          CmHuffman *distance,
                          char *error,
                          size_t error_size)
{
    unsigned char literal_lengths[288];
    unsigned char distance_lengths[32];
    size_t i;
    for (i = 0u; i <= 143u; ++i) literal_lengths[i] = 8u;
    for (; i <= 255u; ++i) literal_lengths[i] = 9u;
    for (; i <= 279u; ++i) literal_lengths[i] = 7u;
    for (; i <= 287u; ++i) literal_lengths[i] = 8u;
    memset(distance_lengths, 5, sizeof(distance_lengths));
    return cm_huffman_build(literal, literal_lengths, 288u, error, error_size) &&
           cm_huffman_build(distance, distance_lengths, 32u, error, error_size);
}

static int cm_dynamic_trees(CmBitReader *reader,
                            CmHuffman *literal,
                            CmHuffman *distance,
                            char *error,
                            size_t error_size)
{
    static const unsigned char order[19] = {
        16u, 17u, 18u, 0u, 8u, 7u, 9u, 6u, 10u, 5u,
        11u, 4u, 12u, 3u, 13u, 2u, 14u, 1u, 15u
    };
    unsigned hlit_bits, hdist_bits, hclen_bits;
    size_t hlit, hdist, hclen, total, at;
    unsigned char code_lengths[19];
    unsigned char lengths[288 + 32];
    CmHuffman code_tree;
    size_t i;

    if (!cm_bits(reader, 5u, &hlit_bits) ||
        !cm_bits(reader, 5u, &hdist_bits) ||
        !cm_bits(reader, 4u, &hclen_bits)) {
        cm_set_error(error, error_size, "truncated DEFLATE dynamic header");
        return 0;
    }
    hlit = (size_t)hlit_bits + 257u;
    hdist = (size_t)hdist_bits + 1u;
    hclen = (size_t)hclen_bits + 4u;
    if (hlit > 286u || hdist > 32u) {
        cm_set_error(error, error_size, "invalid DEFLATE dynamic tree sizes");
        return 0;
    }
    memset(code_lengths, 0, sizeof(code_lengths));
    for (i = 0u; i < hclen; ++i) {
        unsigned value;
        if (!cm_bits(reader, 3u, &value)) {
            cm_set_error(error, error_size, "truncated DEFLATE code-length tree");
            return 0;
        }
        code_lengths[order[i]] = (unsigned char)value;
    }
    if (!cm_huffman_build(&code_tree, code_lengths, 19u, error, error_size) ||
        code_tree.empty) {
        if (code_tree.empty) {
            cm_set_error(error, error_size, "empty DEFLATE code-length tree");
        }
        return 0;
    }
    total = hlit + hdist;
    at = 0u;
    while (at < total) {
        unsigned symbol;
        if (!cm_huffman_decode(reader, &code_tree, &symbol)) {
            cm_set_error(error, error_size, "invalid DEFLATE code-length symbol");
            return 0;
        }
        if (symbol <= 15u) {
            lengths[at++] = (unsigned char)symbol;
        } else if (symbol == 16u) {
            unsigned extra;
            size_t repeat;
            unsigned char previous;
            if (at == 0u || !cm_bits(reader, 2u, &extra)) {
                cm_set_error(error, error_size, "invalid DEFLATE repeat code 16");
                return 0;
            }
            repeat = (size_t)extra + 3u;
            if (repeat > total - at) {
                cm_set_error(error, error_size, "DEFLATE code-length repeat overruns tree");
                return 0;
            }
            previous = lengths[at - 1u];
            while (repeat-- != 0u) lengths[at++] = previous;
        } else if (symbol == 17u || symbol == 18u) {
            unsigned extra;
            unsigned extra_count = symbol == 17u ? 3u : 7u;
            size_t repeat;
            if (!cm_bits(reader, extra_count, &extra)) {
                cm_set_error(error, error_size, "truncated DEFLATE zero repeat");
                return 0;
            }
            repeat = (size_t)extra + (symbol == 17u ? 3u : 11u);
            if (repeat > total - at) {
                cm_set_error(error, error_size, "DEFLATE zero repeat overruns tree");
                return 0;
            }
            while (repeat-- != 0u) lengths[at++] = 0u;
        } else {
            cm_set_error(error, error_size, "invalid DEFLATE code-length symbol %u", symbol);
            return 0;
        }
    }
    if (lengths[256] == 0u) {
        cm_set_error(error, error_size, "DEFLATE literal tree has no end-of-block code");
        return 0;
    }
    return cm_huffman_build(literal, lengths, hlit, error, error_size) &&
           cm_huffman_build(distance, lengths + hlit, hdist, error, error_size);
}

static int cm_inflate_compressed_block(CmBitReader *reader,
                                       const CmHuffman *literal,
                                       const CmHuffman *distance,
                                       unsigned char *output,
                                       size_t output_size,
                                       size_t *output_pos,
                                       char *error,
                                       size_t error_size)
{
    static const uint16_t length_base[29] = {
        3u, 4u, 5u, 6u, 7u, 8u, 9u, 10u, 11u, 13u,
        15u, 17u, 19u, 23u, 27u, 31u, 35u, 43u, 51u, 59u,
        67u, 83u, 99u, 115u, 131u, 163u, 195u, 227u, 258u
    };
    static const unsigned char length_extra[29] = {
        0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 1u, 1u,
        1u, 1u, 2u, 2u, 2u, 2u, 3u, 3u, 3u, 3u,
        4u, 4u, 4u, 4u, 5u, 5u, 5u, 5u, 0u
    };
    static const uint16_t distance_base[30] = {
        1u, 2u, 3u, 4u, 5u, 7u, 9u, 13u, 17u, 25u,
        33u, 49u, 65u, 97u, 129u, 193u, 257u, 385u, 513u, 769u,
        1025u, 1537u, 2049u, 3073u, 4097u, 6145u, 8193u, 12289u,
        16385u, 24577u
    };
    static const unsigned char distance_extra[30] = {
        0u, 0u, 0u, 0u, 1u, 1u, 2u, 2u, 3u, 3u,
        4u, 4u, 5u, 5u, 6u, 6u, 7u, 7u, 8u, 8u,
        9u, 9u, 10u, 10u, 11u, 11u, 12u, 12u, 13u, 13u
    };

    for (;;) {
        unsigned symbol;
        if (!cm_huffman_decode(reader, literal, &symbol)) {
            cm_set_error(error, error_size, "invalid or truncated DEFLATE literal code");
            return 0;
        }
        if (symbol < 256u) {
            if (*output_pos >= output_size) {
                cm_set_error(error, error_size, "DEFLATE output exceeds declared ZIP size");
                return 0;
            }
            output[(*output_pos)++] = (unsigned char)symbol;
        } else if (symbol == 256u) {
            return 1;
        } else if (symbol <= 285u) {
            unsigned length_index = symbol - 257u;
            unsigned extra = 0u;
            size_t length;
            unsigned distance_symbol;
            size_t distance_value;
            size_t i;
            if (length_extra[length_index] != 0u &&
                !cm_bits(reader, length_extra[length_index], &extra)) {
                cm_set_error(error, error_size, "truncated DEFLATE match length");
                return 0;
            }
            length = (size_t)length_base[length_index] + extra;
            if (!cm_huffman_decode(reader, distance, &distance_symbol) ||
                distance_symbol >= 30u) {
                cm_set_error(error, error_size, "invalid DEFLATE distance code");
                return 0;
            }
            extra = 0u;
            if (distance_extra[distance_symbol] != 0u &&
                !cm_bits(reader, distance_extra[distance_symbol], &extra)) {
                cm_set_error(error, error_size, "truncated DEFLATE distance");
                return 0;
            }
            distance_value = (size_t)distance_base[distance_symbol] + extra;
            if (distance_value == 0u || distance_value > *output_pos) {
                cm_set_error(error, error_size, "DEFLATE match refers before output start");
                return 0;
            }
            if (length > output_size - *output_pos) {
                cm_set_error(error, error_size, "DEFLATE match exceeds declared ZIP size");
                return 0;
            }
            for (i = 0u; i < length; ++i) {
                output[*output_pos] = output[*output_pos - distance_value];
                ++*output_pos;
            }
        } else {
            cm_set_error(error, error_size, "reserved DEFLATE literal symbol %u", symbol);
            return 0;
        }
    }
}

static int cm_inflate_raw(const unsigned char *input,
                          size_t input_size,
                          unsigned char *output,
                          size_t output_size,
                          char *error,
                          size_t error_size)
{
    CmBitReader reader;
    size_t output_pos = 0u;
    int final_block = 0;

    reader.data = input;
    reader.size = input_size;
    reader.byte_pos = 0u;
    reader.bit_pos = 0u;

    while (!final_block) {
        unsigned final_value, block_type;
        if (!cm_bits(&reader, 1u, &final_value) || !cm_bits(&reader, 2u, &block_type)) {
            cm_set_error(error, error_size, "truncated DEFLATE block header");
            return 0;
        }
        final_block = (int)final_value;
        if (block_type == 0u) {
            uint16_t length, complement;
            cm_bit_align_byte(&reader);
            if (!cm_range_ok(reader.byte_pos, 4u, reader.size)) {
                cm_set_error(error, error_size, "truncated DEFLATE stored block header");
                return 0;
            }
            length = cm_get_u16le(reader.data + reader.byte_pos);
            complement = cm_get_u16le(reader.data + reader.byte_pos + 2u);
            reader.byte_pos += 4u;
            if ((uint16_t)(length ^ UINT16_C(0xffff)) != complement) {
                cm_set_error(error, error_size, "invalid DEFLATE stored block length");
                return 0;
            }
            if (!cm_range_ok(reader.byte_pos, length, reader.size) ||
                (size_t)length > output_size - output_pos) {
                cm_set_error(error, error_size, "DEFLATE stored block is truncated or oversized");
                return 0;
            }
            memcpy(output + output_pos, reader.data + reader.byte_pos, length);
            output_pos += length;
            reader.byte_pos += length;
        } else if (block_type == 1u || block_type == 2u) {
            CmHuffman literal, distance;
            int ok = block_type == 1u
                         ? cm_fixed_trees(&literal, &distance, error, error_size)
                         : cm_dynamic_trees(&reader, &literal, &distance, error, error_size);
            if (!ok || !cm_inflate_compressed_block(&reader, &literal, &distance,
                                                     output, output_size, &output_pos,
                                                     error, error_size)) {
                return 0;
            }
        } else {
            cm_set_error(error, error_size, "reserved DEFLATE block type");
            return 0;
        }
    }
    if (output_pos != output_size) {
        cm_set_error(error, error_size,
                     "DEFLATE size mismatch (expected %zu, produced %zu)",
                     output_size, output_pos);
        return 0;
    }
    return 1;
}

typedef struct CmZipEntry {
    uint16_t flags;
    uint16_t method;
    uint32_t crc32;
    uint32_t compressed_size;
    uint32_t uncompressed_size;
    uint32_t local_offset;
    int present;
} CmZipEntry;

static int cm_member_index(const unsigned char *name, size_t length)
{
    size_t i;
    for (i = 0u; i < CM_MEMBER_COUNT; ++i) {
        size_t wanted = strlen(cm_member_names[i]);
        if (wanted == length && memcmp(name, cm_member_names[i], length) == 0) {
            return (int)i;
        }
    }
    return -1;
}

static int cm_zip_catalog(const unsigned char *archive,
                          size_t archive_size,
                          CmZipEntry entries[CM_MEMBER_COUNT],
                          char *error,
                          size_t error_size)
{
    size_t search_start, position, eocd = SIZE_MAX;
    uint16_t disk_number, central_disk, disk_entries, total_entries;
    uint32_t central_size32, central_offset32;
    size_t central_size, central_offset, cursor;
    size_t entry_number;

    memset(entries, 0, sizeof(CmZipEntry) * CM_MEMBER_COUNT);
    if (archive_size < 22u) {
        cm_set_error(error, error_size, "file is too short to be a ZIP archive");
        return 0;
    }
    search_start = archive_size > 65557u ? archive_size - 65557u : 0u;
    position = archive_size - 22u;
    for (;;) {
        if (cm_get_u32le(archive + position) == CM_ZIP_EOCD_SIG &&
            cm_range_ok(position, 22u, archive_size)) {
            uint16_t comment_length = cm_get_u16le(archive + position + 20u);
            if (cm_range_ok(position + 22u, comment_length, archive_size) &&
                position + 22u + comment_length == archive_size) {
                eocd = position;
                break;
            }
        }
        if (position == search_start) break;
        --position;
    }
    if (eocd == SIZE_MAX) {
        cm_set_error(error, error_size, "ZIP end-of-central-directory record not found");
        return 0;
    }
    disk_number = cm_get_u16le(archive + eocd + 4u);
    central_disk = cm_get_u16le(archive + eocd + 6u);
    disk_entries = cm_get_u16le(archive + eocd + 8u);
    total_entries = cm_get_u16le(archive + eocd + 10u);
    central_size32 = cm_get_u32le(archive + eocd + 12u);
    central_offset32 = cm_get_u32le(archive + eocd + 16u);
    if (disk_number != 0u || central_disk != 0u || disk_entries != total_entries) {
        cm_set_error(error, error_size, "multi-disk ZIP archives are not supported");
        return 0;
    }
    if (total_entries == UINT16_MAX || central_size32 == UINT32_MAX ||
        central_offset32 == UINT32_MAX) {
        cm_set_error(error, error_size, "ZIP64 archives are not supported");
        return 0;
    }
    central_size = central_size32;
    central_offset = central_offset32;
    if (!cm_range_ok(central_offset, central_size, archive_size) ||
        central_offset + central_size > eocd) {
        cm_set_error(error, error_size, "invalid ZIP central-directory bounds");
        return 0;
    }
    cursor = central_offset;
    for (entry_number = 0u; entry_number < total_entries; ++entry_number) {
        uint16_t name_length, extra_length, comment_length;
        size_t record_size;
        int index;
        if (!cm_range_ok(cursor, 46u, archive_size) ||
            cm_get_u32le(archive + cursor) != CM_ZIP_CENTRAL_SIG) {
            cm_set_error(error, error_size, "invalid ZIP central-directory entry %zu", entry_number);
            return 0;
        }
        name_length = cm_get_u16le(archive + cursor + 28u);
        extra_length = cm_get_u16le(archive + cursor + 30u);
        comment_length = cm_get_u16le(archive + cursor + 32u);
        record_size = 46u;
        if (!cm_add_size(record_size, name_length, &record_size) ||
            !cm_add_size(record_size, extra_length, &record_size) ||
            !cm_add_size(record_size, comment_length, &record_size) ||
            !cm_range_ok(cursor, record_size, archive_size)) {
            cm_set_error(error, error_size, "truncated ZIP central-directory entry");
            return 0;
        }
        index = cm_member_index(archive + cursor + 46u, name_length);
        if (index >= 0) {
            CmZipEntry *entry = &entries[(size_t)index];
            if (entry->present) {
                cm_set_error(error, error_size, "duplicate ZIP member '%s'", cm_member_names[index]);
                return 0;
            }
            entry->flags = cm_get_u16le(archive + cursor + 8u);
            entry->method = cm_get_u16le(archive + cursor + 10u);
            entry->crc32 = cm_get_u32le(archive + cursor + 16u);
            entry->compressed_size = cm_get_u32le(archive + cursor + 20u);
            entry->uncompressed_size = cm_get_u32le(archive + cursor + 24u);
            entry->local_offset = cm_get_u32le(archive + cursor + 42u);
            entry->present = 1;
            if ((entry->flags & 1u) != 0u) {
                cm_set_error(error, error_size, "encrypted ZIP member '%s' is unsupported",
                             cm_member_names[index]);
                return 0;
            }
            if (entry->method != 0u && entry->method != 8u) {
                cm_set_error(error, error_size,
                             "ZIP member '%s' uses unsupported compression method %u",
                             cm_member_names[index], (unsigned)entry->method);
                return 0;
            }
        }
        cursor += record_size;
    }
    if (cursor != central_offset + central_size) {
        cm_set_error(error, error_size, "ZIP central-directory size mismatch");
        return 0;
    }
    for (entry_number = 0u; entry_number < CM_MEMBER_COUNT; ++entry_number) {
        if (!entries[entry_number].present) {
            cm_set_error(error, error_size, "required ZIP member '%s' is missing",
                         cm_member_names[entry_number]);
            return 0;
        }
    }
    return 1;
}

static int cm_zip_extract(const unsigned char *archive,
                          size_t archive_size,
                          const CmZipEntry *entry,
                          const char *expected_name,
                          unsigned char **output,
                          size_t *output_size,
                          char *error,
                          size_t error_size)
{
    size_t local = entry->local_offset;
    uint16_t local_flags, local_method, name_length, extra_length;
    size_t data_offset;
    unsigned char *result;
    uint32_t crc;

    *output = NULL;
    *output_size = 0u;
    if (!cm_range_ok(local, 30u, archive_size) ||
        cm_get_u32le(archive + local) != CM_ZIP_LOCAL_SIG) {
        cm_set_error(error, error_size, "invalid local ZIP header for '%s'", expected_name);
        return 0;
    }
    local_flags = cm_get_u16le(archive + local + 6u);
    local_method = cm_get_u16le(archive + local + 8u);
    name_length = cm_get_u16le(archive + local + 26u);
    extra_length = cm_get_u16le(archive + local + 28u);
    if (local_flags != entry->flags || local_method != entry->method) {
        cm_set_error(error, error_size, "local/central ZIP metadata disagree for '%s'", expected_name);
        return 0;
    }
    data_offset = local + 30u;
    if (!cm_range_ok(data_offset, name_length, archive_size) ||
        strlen(expected_name) != name_length ||
        memcmp(archive + data_offset, expected_name, name_length) != 0) {
        cm_set_error(error, error_size, "unexpected local ZIP filename for '%s'", expected_name);
        return 0;
    }
    if (!cm_add_size(data_offset, name_length, &data_offset) ||
        !cm_add_size(data_offset, extra_length, &data_offset) ||
        !cm_range_ok(data_offset, entry->compressed_size, archive_size)) {
        cm_set_error(error, error_size, "ZIP member '%s' data is out of bounds", expected_name);
        return 0;
    }
    result = (unsigned char *)malloc(entry->uncompressed_size == 0u
                                         ? 1u
                                         : (size_t)entry->uncompressed_size);
    if (result == NULL) {
        cm_set_error(error, error_size, "out of memory extracting '%s'", expected_name);
        return 0;
    }
    if (entry->method == 0u) {
        if (entry->compressed_size != entry->uncompressed_size) {
            cm_set_error(error, error_size, "stored ZIP member '%s' has inconsistent sizes",
                         expected_name);
            free(result);
            return 0;
        }
        memcpy(result, archive + data_offset, entry->uncompressed_size);
    } else if (!cm_inflate_raw(archive + data_offset, entry->compressed_size,
                               result, entry->uncompressed_size, error, error_size)) {
        free(result);
        return 0;
    }
    crc = cm_crc32(result, entry->uncompressed_size);
    if (crc != entry->crc32) {
        cm_set_error(error, error_size,
                     "CRC-32 mismatch for ZIP member '%s' (expected %08x, got %08x)",
                     expected_name, (unsigned)entry->crc32, (unsigned)crc);
        free(result);
        return 0;
    }
    *output = result;
    *output_size = entry->uncompressed_size;
    return 1;
}

typedef struct CmNpyView {
    char kind;
    unsigned item_size;
    size_t rank;
    size_t dims[8];
    size_t count;
    const unsigned char *data;
} CmNpyView;

static char *cm_npy_find_key(char *header, const char *key)
{
    char pattern[48];
    char *found;
    int written = snprintf(pattern, sizeof(pattern), "'%s'", key);
    if (written <= 0 || (size_t)written >= sizeof(pattern)) {
        return NULL;
    }
    found = strstr(header, pattern);
    if (found != NULL) {
        return found + strlen(pattern);
    }
    written = snprintf(pattern, sizeof(pattern), "\"%s\"", key);
    if (written <= 0 || (size_t)written >= sizeof(pattern)) {
        return NULL;
    }
    found = strstr(header, pattern);
    return found == NULL ? NULL : found + strlen(pattern);
}

static char *cm_skip_space(char *p)
{
    while (*p == ' ' || *p == '\t' || *p == '\r' || *p == '\n') {
        ++p;
    }
    return p;
}

static int cm_npy_after_colon(char **cursor)
{
    char *p = cm_skip_space(*cursor);
    if (*p != ':') {
        return 0;
    }
    *cursor = cm_skip_space(p + 1);
    return 1;
}

static int cm_parse_decimal_size(char **cursor, size_t *value)
{
    char *p = *cursor;
    size_t result = 0u;
    if (*p < '0' || *p > '9') {
        return 0;
    }
    do {
        unsigned digit = (unsigned)(*p - '0');
        if (result > (SIZE_MAX - digit) / 10u) {
            return 0;
        }
        result = result * 10u + digit;
        ++p;
    } while (*p >= '0' && *p <= '9');
    *cursor = p;
    *value = result;
    return 1;
}

static int cm_npy_parse(const unsigned char *buffer,
                        size_t buffer_size,
                        const char *member_name,
                        CmNpyView *view,
                        char *error,
                        size_t error_size)
{
    static const unsigned char magic[6] = { 0x93u, 'N', 'U', 'M', 'P', 'Y' };
    uint16_t header_length;
    size_t data_offset, expected_bytes;
    char *header = NULL;
    char *p;
    char quote, endian;
    size_t dtype_length;
    size_t i;

    memset(view, 0, sizeof(*view));
    if (buffer_size < CM_NPY_PREFIX_SIZE || memcmp(buffer, magic, sizeof(magic)) != 0) {
        cm_set_error(error, error_size, "ZIP member '%s' is not an NPY file", member_name);
        return 0;
    }
    if (buffer[6] != 1u || buffer[7] != 0u) {
        cm_set_error(error, error_size,
                     "ZIP member '%s' uses unsupported NPY version %u.%u (need 1.0)",
                     member_name, (unsigned)buffer[6], (unsigned)buffer[7]);
        return 0;
    }
    header_length = cm_get_u16le(buffer + 8u);
    data_offset = CM_NPY_PREFIX_SIZE + (size_t)header_length;
    if (!cm_range_ok(CM_NPY_PREFIX_SIZE, header_length, buffer_size)) {
        cm_set_error(error, error_size, "truncated NPY header in '%s'", member_name);
        return 0;
    }
    header = (char *)malloc((size_t)header_length + 1u);
    if (header == NULL) {
        cm_set_error(error, error_size, "out of memory parsing '%s'", member_name);
        return 0;
    }
    memcpy(header, buffer + CM_NPY_PREFIX_SIZE, header_length);
    header[header_length] = '\0';
    if (memchr(header, '\0', header_length) != NULL) {
        cm_set_error(error, error_size, "NUL byte in NPY header for '%s'", member_name);
        free(header);
        return 0;
    }

    p = cm_npy_find_key(header, "descr");
    if (p == NULL || !cm_npy_after_colon(&p) || (*p != '\'' && *p != '\"')) {
        cm_set_error(error, error_size, "missing NPY dtype descriptor in '%s'", member_name);
        free(header);
        return 0;
    }
    quote = *p++;
    {
        char *end = strchr(p, quote);
        if (end == NULL) {
            cm_set_error(error, error_size, "unterminated NPY dtype descriptor in '%s'", member_name);
            free(header);
            return 0;
        }
        dtype_length = (size_t)(end - p);
    }
    if (dtype_length != 3u ||
        (p[0] != '<' && p[0] != '|') ||
        (p[1] != 'f' && p[1] != 'i') ||
        (p[2] != '4' && p[2] != '8')) {
        cm_set_error(error, error_size, "unsupported NPY dtype '%.*s' in '%s'",
                     (int)dtype_length, p, member_name);
        free(header);
        return 0;
    }
    endian = p[0];
    view->kind = p[1];
    view->item_size = (unsigned)(p[2] - '0');
    if (endian == '|' && view->item_size != 1u) {
        cm_set_error(error, error_size, "non-endian NPY marker is invalid for '%s'", member_name);
        free(header);
        return 0;
    }
    if ((view->kind == 'f' && view->item_size != 8u) ||
        (view->kind == 'i' && view->item_size != 4u && view->item_size != 8u)) {
        cm_set_error(error, error_size, "unsupported NPY scalar type in '%s'", member_name);
        free(header);
        return 0;
    }

    p = cm_npy_find_key(header, "fortran_order");
    if (p == NULL || !cm_npy_after_colon(&p) || strncmp(p, "False", 5u) != 0) {
        cm_set_error(error, error_size, "NPY member '%s' is not C-order", member_name);
        free(header);
        return 0;
    }

    p = cm_npy_find_key(header, "shape");
    if (p == NULL || !cm_npy_after_colon(&p) || *p != '(') {
        cm_set_error(error, error_size, "missing NPY shape in '%s'", member_name);
        free(header);
        return 0;
    }
    p = cm_skip_space(p + 1);
    if (*p == ')') {
        ++p;
        view->rank = 0u;
    } else {
        for (;;) {
            size_t dimension;
            if (view->rank >= sizeof(view->dims) / sizeof(view->dims[0]) ||
                !cm_parse_decimal_size(&p, &dimension)) {
                cm_set_error(error, error_size, "invalid or over-rank NPY shape in '%s'", member_name);
                free(header);
                return 0;
            }
            view->dims[view->rank++] = dimension;
            p = cm_skip_space(p);
            if (*p == ',') {
                p = cm_skip_space(p + 1);
                if (*p == ')') {
                    ++p;
                    break;
                }
            } else if (*p == ')') {
                ++p;
                break;
            } else {
                cm_set_error(error, error_size, "malformed NPY shape in '%s'", member_name);
                free(header);
                return 0;
            }
        }
    }
    view->count = 1u;
    for (i = 0u; i < view->rank; ++i) {
        if (!cm_mul_size(view->count, view->dims[i], &view->count)) {
            cm_set_error(error, error_size, "NPY shape overflows address space in '%s'", member_name);
            free(header);
            return 0;
        }
    }
    if (!cm_mul_size(view->count, view->item_size, &expected_bytes) ||
        !cm_range_ok(data_offset, expected_bytes, buffer_size) ||
        data_offset + expected_bytes != buffer_size) {
        cm_set_error(error, error_size, "NPY payload size does not match shape in '%s'", member_name);
        free(header);
        return 0;
    }
    view->data = buffer + data_offset;
    free(header);
    return 1;
}

static double cm_npy_f64_at(const CmNpyView *view, size_t index)
{
    uint64_t bits = cm_get_u64le(view->data + index * 8u);
    double value;
    memcpy(&value, &bits, sizeof(value));
    return value;
}

static int64_t cm_npy_i64_at(const CmNpyView *view, size_t index)
{
    if (view->item_size == 4u) {
        uint32_t bits = cm_get_u32le(view->data + index * 4u);
        int32_t value;
        memcpy(&value, &bits, sizeof(value));
        return value;
    } else {
        uint64_t bits = cm_get_u64le(view->data + index * 8u);
        int64_t value;
        memcpy(&value, &bits, sizeof(value));
        return value;
    }
}

void cm_dataset_init(DatasetCollection *dataset)
{
    if (dataset != NULL) {
        memset(dataset, 0, sizeof(*dataset));
    }
}

void cm_dataset_free(DatasetCollection *dataset)
{
    if (dataset == NULL) {
        return;
    }
    free(dataset->X);
    free(dataset->Y);
    free(dataset->g);
    free(dataset->seeds);
    cm_dataset_init(dataset);
}

int cm_dataset_alloc(DatasetCollection *dataset,
                     size_t n_sim,
                     size_t n_pop,
                     size_t n_cov,
                     char *error,
                     size_t error_size)
{
    DatasetCollection temporary;
    size_t simulation_units, x_count;

    if (error != NULL && error_size != 0u) error[0] = '\0';
    if (dataset == NULL) {
        cm_set_error(error, error_size, "dataset output pointer is NULL");
        return 0;
    }
    if (n_sim == 0u || n_pop == 0u || n_cov == 0u) {
        cm_set_error(error, error_size, "dataset dimensions must all be positive");
        return 0;
    }
    if (!cm_mul_size(n_sim, n_pop, &simulation_units) ||
        !cm_mul_size(simulation_units, n_cov, &x_count) ||
        x_count > SIZE_MAX / sizeof(double) ||
        simulation_units > SIZE_MAX / sizeof(double) ||
        simulation_units > SIZE_MAX / sizeof(int) ||
        n_sim > SIZE_MAX / sizeof(uint64_t)) {
        cm_set_error(error, error_size, "dataset dimensions overflow address space");
        return 0;
    }
    cm_dataset_init(&temporary);
    temporary.X = (double *)calloc(x_count, sizeof(double));
    temporary.Y = (double *)calloc(simulation_units, sizeof(double));
    temporary.g = (int *)calloc(simulation_units, sizeof(int));
    temporary.seeds = (uint64_t *)calloc(n_sim, sizeof(uint64_t));
    if (temporary.X == NULL || temporary.Y == NULL ||
        temporary.g == NULL || temporary.seeds == NULL) {
        cm_set_error(error, error_size, "out of memory allocating dataset arrays");
        cm_dataset_free(&temporary);
        return 0;
    }
    temporary.n_sim = n_sim;
    temporary.n_pop = n_pop;
    temporary.n_cov = n_cov;
    *dataset = temporary;
    return 1;
}

static int cm_npy_require(const CmNpyView *view,
                          char kind,
                          unsigned item_size,
                          size_t rank,
                          const char *name,
                          char *error,
                          size_t error_size)
{
    if (view->kind != kind ||
        (item_size != 0u && view->item_size != item_size) ||
        view->rank != rank) {
        cm_set_error(error, error_size,
                     "NPY member '%s' has wrong dtype or rank", name);
        return 0;
    }
    return 1;
}

int cm_dataset_load_npz(const char *path,
                        DatasetCollection *out,
                        char *error,
                        size_t error_size)
{
    unsigned char *archive = NULL;
    size_t archive_size = 0u;
    CmZipEntry entries[CM_MEMBER_COUNT];
    unsigned char *members[CM_MEMBER_COUNT] = { NULL };
    size_t member_sizes[CM_MEMBER_COUNT] = { 0u };
    CmNpyView views[CM_MEMBER_COUNT];
    DatasetCollection loaded;
    int64_t n_sim_value, n_pop_value;
    size_t n_sim, n_pop, n_cov, unit_count, x_count;
    size_t i;
    int success = 0;

    if (error != NULL && error_size != 0u) error[0] = '\0';
    if (out == NULL) {
        cm_set_error(error, error_size, "dataset output pointer is NULL");
        return 0;
    }
    cm_dataset_init(out);
    cm_dataset_init(&loaded);
    if (path == NULL || *path == '\0') {
        cm_set_error(error, error_size, "dataset path is empty");
        return 0;
    }
    if (sizeof(double) != 8u || sizeof(int32_t) != 4u || sizeof(int64_t) != 8u) {
        cm_set_error(error, error_size, "this platform does not provide required 32/64-bit types");
        return 0;
    }
    if (!cm_read_file(path, &archive, &archive_size, error, error_size) ||
        !cm_zip_catalog(archive, archive_size, entries, error, error_size)) {
        goto cleanup;
    }
    for (i = 0u; i < CM_MEMBER_COUNT; ++i) {
        if (!cm_zip_extract(archive, archive_size, &entries[i], cm_member_names[i],
                            &members[i], &member_sizes[i], error, error_size) ||
            !cm_npy_parse(members[i], member_sizes[i], cm_member_names[i], &views[i],
                          error, error_size)) {
            goto cleanup;
        }
    }
    if (!cm_npy_require(&views[0], 'f', 8u, 3u, "X.npy", error, error_size) ||
        !cm_npy_require(&views[1], 'f', 8u, 2u, "Y.npy", error, error_size) ||
        !cm_npy_require(&views[2], 'i', 0u, 2u, "g.npy", error, error_size) ||
        !cm_npy_require(&views[3], 'i', 0u, 1u, "seeds.npy", error, error_size) ||
        !cm_npy_require(&views[4], 'i', 0u, 0u, "n_sim.npy", error, error_size) ||
        !cm_npy_require(&views[5], 'i', 0u, 0u, "n_pop.npy", error, error_size) ||
        !cm_npy_require(&views[6], 'f', 8u, 0u, "perc_rc.npy", error, error_size) ||
        !cm_npy_require(&views[7], 'f', 8u, 0u, "sb.npy", error, error_size)) {
        goto cleanup;
    }
    if ((views[2].item_size != 4u && views[2].item_size != 8u) ||
        (views[3].item_size != 4u && views[3].item_size != 8u) ||
        (views[4].item_size != 4u && views[4].item_size != 8u) ||
        (views[5].item_size != 4u && views[5].item_size != 8u)) {
        cm_set_error(error, error_size, "integer NPY members must be int32 or int64");
        goto cleanup;
    }
    n_sim_value = cm_npy_i64_at(&views[4], 0u);
    n_pop_value = cm_npy_i64_at(&views[5], 0u);
    if (n_sim_value <= 0 || n_pop_value <= 0 ||
        (uint64_t)n_sim_value > (uint64_t)SIZE_MAX ||
        (uint64_t)n_pop_value > (uint64_t)SIZE_MAX) {
        cm_set_error(error, error_size, "n_sim and n_pop metadata must be positive and addressable");
        goto cleanup;
    }
    n_sim = (size_t)n_sim_value;
    n_pop = (size_t)n_pop_value;
    n_cov = views[0].dims[2];
    if (n_cov == 0u || views[0].dims[0] != n_sim || views[0].dims[1] != n_pop ||
        views[1].dims[0] != n_sim || views[1].dims[1] != n_pop ||
        views[2].dims[0] != n_sim || views[2].dims[1] != n_pop ||
        views[3].dims[0] != n_sim) {
        cm_set_error(error, error_size,
                     "X, Y, g, seeds, n_sim, and n_pop dimensions are inconsistent");
        goto cleanup;
    }
    if (!cm_mul_size(n_sim, n_pop, &unit_count) ||
        !cm_mul_size(unit_count, n_cov, &x_count)) {
        cm_set_error(error, error_size, "dataset dimensions overflow address space");
        goto cleanup;
    }
    if (!cm_dataset_alloc(&loaded, n_sim, n_pop, n_cov, error, error_size)) {
        goto cleanup;
    }
    loaded.perc_rc = cm_npy_f64_at(&views[6], 0u);
    loaded.sb = cm_npy_f64_at(&views[7], 0u);
    if (!isfinite(loaded.perc_rc) || loaded.perc_rc < 0.0 || loaded.perc_rc > 1.0 ||
        !isfinite(loaded.sb)) {
        cm_set_error(error, error_size, "perc_rc/sb metadata are invalid");
        goto cleanup;
    }
    for (i = 0u; i < x_count; ++i) loaded.X[i] = cm_npy_f64_at(&views[0], i);
    for (i = 0u; i < unit_count; ++i) loaded.Y[i] = cm_npy_f64_at(&views[1], i);
    for (i = 0u; i < unit_count; ++i) {
        int64_t value = cm_npy_i64_at(&views[2], i);
        if (value < INT_MIN || value > INT_MAX) {
            cm_set_error(error, error_size, "g value at index %zu does not fit C int", i);
            goto cleanup;
        }
        loaded.g[i] = (int)value;
    }
    for (i = 0u; i < n_sim; ++i) {
        int64_t value = cm_npy_i64_at(&views[3], i);
        if (value < 0) {
            cm_set_error(error, error_size, "negative seed at index %zu is unsupported", i);
            goto cleanup;
        }
        loaded.seeds[i] = (uint64_t)value;
    }
    *out = loaded;
    cm_dataset_init(&loaded);
    success = 1;

cleanup:
    cm_dataset_free(&loaded);
    for (i = 0u; i < CM_MEMBER_COUNT; ++i) free(members[i]);
    free(archive);
    return success;
}

typedef struct CmOwnedNpy {
    unsigned char *buffer;
    size_t size;
    uint32_t crc32;
} CmOwnedNpy;

static int cm_append_format(char *buffer,
                            size_t capacity,
                            size_t *length,
                            const char *format,
                            ...)
{
    va_list args;
    int written;
    if (*length >= capacity) return 0;
    va_start(args, format);
    written = vsnprintf(buffer + *length, capacity - *length, format, args);
    va_end(args);
    if (written < 0 || (size_t)written >= capacity - *length) {
        return 0;
    }
    *length += (size_t)written;
    return 1;
}

static int cm_npy_create(const char *descr,
                         size_t rank,
                         const size_t *dims,
                         CmOwnedNpy *owned,
                         unsigned char **payload,
                         char *error,
                         size_t error_size)
{
    static const unsigned char magic[6] = { 0x93u, 'N', 'U', 'M', 'P', 'Y' };
    char shape[192];
    char dictionary[320];
    size_t shape_length = 0u;
    size_t dictionary_length;
    size_t count = 1u;
    size_t item_size;
    size_t payload_size;
    size_t padding;
    size_t header_length;
    size_t total_size;
    size_t i;

    owned->buffer = NULL;
    owned->size = 0u;
    owned->crc32 = 0u;
    *payload = NULL;
    if (strcmp(descr, "<f8") == 0 || strcmp(descr, "<i8") == 0) {
        item_size = 8u;
    } else if (strcmp(descr, "<i4") == 0) {
        item_size = 4u;
    } else {
        cm_set_error(error, error_size, "internal error: unsupported NPY output dtype");
        return 0;
    }
    if (!cm_append_format(shape, sizeof(shape), &shape_length, "(")) {
        cm_set_error(error, error_size, "internal error constructing NPY shape");
        return 0;
    }
    for (i = 0u; i < rank; ++i) {
        if (!cm_mul_size(count, dims[i], &count) ||
            !cm_append_format(shape, sizeof(shape), &shape_length,
                              i == 0u ? "%zu" : ", %zu", dims[i])) {
            cm_set_error(error, error_size, "NPY shape is too large");
            return 0;
        }
    }
    if (rank == 1u &&
        !cm_append_format(shape, sizeof(shape), &shape_length, ",")) {
        cm_set_error(error, error_size, "internal error constructing NPY shape");
        return 0;
    }
    if (!cm_append_format(shape, sizeof(shape), &shape_length, ")")) {
        cm_set_error(error, error_size, "internal error constructing NPY shape");
        return 0;
    }
    {
        int written = snprintf(dictionary, sizeof(dictionary),
                               "{'descr': '%s', 'fortran_order': False, 'shape': %s, }",
                               descr, shape);
        if (written < 0 || (size_t)written >= sizeof(dictionary)) {
            cm_set_error(error, error_size, "NPY header is too large");
            return 0;
        }
        dictionary_length = (size_t)written;
    }
    /* NPY v1 requires magic+version+length+header to end on a 16-byte boundary. */
    padding = (16u - ((CM_NPY_PREFIX_SIZE + dictionary_length + 1u) % 16u)) % 16u;
    header_length = dictionary_length + padding + 1u;
    if (header_length > UINT16_MAX ||
        !cm_mul_size(count, item_size, &payload_size) ||
        !cm_add_size(CM_NPY_PREFIX_SIZE, header_length, &total_size) ||
        !cm_add_size(total_size, payload_size, &total_size) ||
        total_size > UINT32_MAX) {
        cm_set_error(error, error_size, "NPY member exceeds classic ZIP/NPY limits");
        return 0;
    }
    owned->buffer = (unsigned char *)calloc(total_size == 0u ? 1u : total_size, 1u);
    if (owned->buffer == NULL) {
        cm_set_error(error, error_size, "out of memory constructing NPY member");
        return 0;
    }
    memcpy(owned->buffer, magic, sizeof(magic));
    owned->buffer[6] = 1u;
    owned->buffer[7] = 0u;
    cm_put_u16le(owned->buffer + 8u, (uint16_t)header_length);
    memcpy(owned->buffer + CM_NPY_PREFIX_SIZE, dictionary, dictionary_length);
    memset(owned->buffer + CM_NPY_PREFIX_SIZE + dictionary_length, ' ', padding);
    owned->buffer[CM_NPY_PREFIX_SIZE + header_length - 1u] = '\n';
    owned->size = total_size;
    *payload = owned->buffer + CM_NPY_PREFIX_SIZE + header_length;
    return 1;
}

static void cm_npy_owned_free(CmOwnedNpy *owned)
{
    free(owned->buffer);
    owned->buffer = NULL;
    owned->size = 0u;
    owned->crc32 = 0u;
}

static void cm_encode_f64(unsigned char *target, double value)
{
    uint64_t bits;
    memcpy(&bits, &value, sizeof(bits));
    cm_put_u64le(target, bits);
}

static void cm_encode_i64(unsigned char *target, int64_t value)
{
    uint64_t bits;
    memcpy(&bits, &value, sizeof(bits));
    cm_put_u64le(target, bits);
}

static void cm_encode_i32(unsigned char *target, int32_t value)
{
    uint32_t bits;
    memcpy(&bits, &value, sizeof(bits));
    cm_put_u32le(target, bits);
}

static int cm_write_all(FILE *file,
                        const void *buffer,
                        size_t size,
                        const char *path,
                        char *error,
                        size_t error_size)
{
    if (size != 0u && fwrite(buffer, 1u, size, file) != size) {
        cm_set_error(error, error_size, "write failed for '%s': %s", path, strerror(errno));
        return 0;
    }
    return 1;
}

int cm_dataset_save_npz(const char *path,
                        const DatasetCollection *dataset,
                        char *error,
                        size_t error_size)
{
    CmOwnedNpy members[CM_MEMBER_COUNT];
    unsigned char *payloads[CM_MEMBER_COUNT] = { NULL };
    size_t x_dims[3], matrix_dims[2], seeds_dims[1];
    size_t unit_count, x_count;
    size_t i;
    size_t archive_offset = 0u;
    uint32_t local_offsets[CM_MEMBER_COUNT];
    size_t central_offset, central_size;
    FILE *file = NULL;
    int success = 0;

    memset(members, 0, sizeof(members));
    if (error != NULL && error_size != 0u) error[0] = '\0';
    if (path == NULL || *path == '\0') {
        cm_set_error(error, error_size, "dataset output path is empty");
        return 0;
    }
    if (dataset == NULL || dataset->n_sim == 0u || dataset->n_pop == 0u ||
        dataset->n_cov == 0u || dataset->X == NULL || dataset->Y == NULL ||
        dataset->g == NULL || dataset->seeds == NULL) {
        cm_set_error(error, error_size, "dataset is missing dimensions or arrays");
        return 0;
    }
    if (sizeof(double) != 8u || dataset->n_sim > (size_t)INT64_MAX ||
        dataset->n_pop > (size_t)INT64_MAX ||
        !isfinite(dataset->perc_rc) || dataset->perc_rc < 0.0 || dataset->perc_rc > 1.0 ||
        !isfinite(dataset->sb) ||
        !cm_mul_size(dataset->n_sim, dataset->n_pop, &unit_count) ||
        !cm_mul_size(unit_count, dataset->n_cov, &x_count)) {
        cm_set_error(error, error_size, "dataset dimensions or metadata are invalid");
        return 0;
    }
    x_dims[0] = dataset->n_sim;
    x_dims[1] = dataset->n_pop;
    x_dims[2] = dataset->n_cov;
    matrix_dims[0] = dataset->n_sim;
    matrix_dims[1] = dataset->n_pop;
    seeds_dims[0] = dataset->n_sim;

    if (!cm_npy_create("<f8", 3u, x_dims, &members[0], &payloads[0], error, error_size) ||
        !cm_npy_create("<f8", 2u, matrix_dims, &members[1], &payloads[1], error, error_size) ||
        !cm_npy_create("<i4", 2u, matrix_dims, &members[2], &payloads[2], error, error_size) ||
        !cm_npy_create("<i8", 1u, seeds_dims, &members[3], &payloads[3], error, error_size) ||
        !cm_npy_create("<i8", 0u, NULL, &members[4], &payloads[4], error, error_size) ||
        !cm_npy_create("<i8", 0u, NULL, &members[5], &payloads[5], error, error_size) ||
        !cm_npy_create("<f8", 0u, NULL, &members[6], &payloads[6], error, error_size) ||
        !cm_npy_create("<f8", 0u, NULL, &members[7], &payloads[7], error, error_size)) {
        goto cleanup;
    }
    for (i = 0u; i < x_count; ++i) cm_encode_f64(payloads[0] + i * 8u, dataset->X[i]);
    for (i = 0u; i < unit_count; ++i) cm_encode_f64(payloads[1] + i * 8u, dataset->Y[i]);
    for (i = 0u; i < unit_count; ++i) {
        if (dataset->g[i] < INT32_MIN || dataset->g[i] > INT32_MAX) {
            cm_set_error(error, error_size, "g value at index %zu does not fit int32", i);
            goto cleanup;
        }
        cm_encode_i32(payloads[2] + i * 4u, (int32_t)dataset->g[i]);
    }
    for (i = 0u; i < dataset->n_sim; ++i) {
        if (dataset->seeds[i] > (uint64_t)INT64_MAX) {
            cm_set_error(error, error_size, "seed at index %zu does not fit NumPy int64", i);
            goto cleanup;
        }
        cm_encode_i64(payloads[3] + i * 8u, (int64_t)dataset->seeds[i]);
    }
    cm_encode_i64(payloads[4], (int64_t)dataset->n_sim);
    cm_encode_i64(payloads[5], (int64_t)dataset->n_pop);
    cm_encode_f64(payloads[6], dataset->perc_rc);
    cm_encode_f64(payloads[7], dataset->sb);
    for (i = 0u; i < CM_MEMBER_COUNT; ++i) {
        size_t local_size = 30u + strlen(cm_member_names[i]) + members[i].size;
        members[i].crc32 = cm_crc32(members[i].buffer, members[i].size);
        if (archive_offset > UINT32_MAX ||
            !cm_add_size(archive_offset, local_size, &archive_offset) ||
            archive_offset > UINT32_MAX) {
            cm_set_error(error, error_size, "NPZ archive exceeds classic ZIP size limits");
            goto cleanup;
        }
    }
    central_offset = archive_offset;
    central_size = 0u;
    for (i = 0u; i < CM_MEMBER_COUNT; ++i) {
        if (!cm_add_size(central_size, 46u + strlen(cm_member_names[i]), &central_size)) {
            cm_set_error(error, error_size, "NPZ central directory size overflow");
            goto cleanup;
        }
    }
    if (central_offset > UINT32_MAX || central_size > UINT32_MAX ||
        !cm_add_size(archive_offset, central_size, &archive_offset) ||
        !cm_add_size(archive_offset, 22u, &archive_offset) || archive_offset > UINT32_MAX) {
        cm_set_error(error, error_size, "NPZ archive exceeds classic ZIP size limits");
        goto cleanup;
    }

    file = fopen(path, "wb");
    if (file == NULL) {
        cm_set_error(error, error_size, "cannot create '%s': %s", path, strerror(errno));
        goto cleanup;
    }
    archive_offset = 0u;
    for (i = 0u; i < CM_MEMBER_COUNT; ++i) {
        unsigned char header[30];
        size_t name_length = strlen(cm_member_names[i]);
        memset(header, 0, sizeof(header));
        cm_put_u32le(header, CM_ZIP_LOCAL_SIG);
        cm_put_u16le(header + 4u, 20u);
        cm_put_u16le(header + 6u, 0u);
        cm_put_u16le(header + 8u, 0u);
        cm_put_u32le(header + 14u, members[i].crc32);
        cm_put_u32le(header + 18u, (uint32_t)members[i].size);
        cm_put_u32le(header + 22u, (uint32_t)members[i].size);
        cm_put_u16le(header + 26u, (uint16_t)name_length);
        local_offsets[i] = (uint32_t)archive_offset;
        if (!cm_write_all(file, header, sizeof(header), path, error, error_size) ||
            !cm_write_all(file, cm_member_names[i], name_length, path, error, error_size) ||
            !cm_write_all(file, members[i].buffer, members[i].size, path, error, error_size)) {
            goto cleanup;
        }
        archive_offset += sizeof(header) + name_length + members[i].size;
    }
    if (archive_offset != central_offset) {
        cm_set_error(error, error_size, "internal ZIP offset mismatch");
        goto cleanup;
    }
    for (i = 0u; i < CM_MEMBER_COUNT; ++i) {
        unsigned char header[46];
        size_t name_length = strlen(cm_member_names[i]);
        memset(header, 0, sizeof(header));
        cm_put_u32le(header, CM_ZIP_CENTRAL_SIG);
        cm_put_u16le(header + 4u, 20u);
        cm_put_u16le(header + 6u, 20u);
        cm_put_u16le(header + 8u, 0u);
        cm_put_u16le(header + 10u, 0u);
        cm_put_u32le(header + 16u, members[i].crc32);
        cm_put_u32le(header + 20u, (uint32_t)members[i].size);
        cm_put_u32le(header + 24u, (uint32_t)members[i].size);
        cm_put_u16le(header + 28u, (uint16_t)name_length);
        cm_put_u32le(header + 42u, local_offsets[i]);
        if (!cm_write_all(file, header, sizeof(header), path, error, error_size) ||
            !cm_write_all(file, cm_member_names[i], name_length, path, error, error_size)) {
            goto cleanup;
        }
        archive_offset += sizeof(header) + name_length;
    }
    {
        unsigned char eocd[22];
        memset(eocd, 0, sizeof(eocd));
        cm_put_u32le(eocd, CM_ZIP_EOCD_SIG);
        cm_put_u16le(eocd + 8u, (uint16_t)CM_MEMBER_COUNT);
        cm_put_u16le(eocd + 10u, (uint16_t)CM_MEMBER_COUNT);
        cm_put_u32le(eocd + 12u, (uint32_t)central_size);
        cm_put_u32le(eocd + 16u, (uint32_t)central_offset);
        if (!cm_write_all(file, eocd, sizeof(eocd), path, error, error_size)) {
            goto cleanup;
        }
    }
    if (fclose(file) != 0) {
        file = NULL;
        cm_set_error(error, error_size, "error closing '%s': %s", path, strerror(errno));
        goto cleanup;
    }
    file = NULL;
    success = 1;

cleanup:
    if (file != NULL) (void)fclose(file);
    for (i = 0u; i < CM_MEMBER_COUNT; ++i) cm_npy_owned_free(&members[i]);
    return success;
}
