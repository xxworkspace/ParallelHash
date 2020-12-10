// Copyright (c) 2020, Bacoo Zhao - zhaoyanbin@bigo.sg/bacoo_zh@163.com
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
// Includes work from parallel-hashmap (https://github.com/greg7mdp/parallel-hashmap)
// with modifications.
// ---------------------------------------------------------------------------

#ifndef _ZMAP_H_
#define _ZMAP_H_

#include <set>
#include <mutex>
#include <atomic>
#include <string>
#include <vector>
#include <thread>
#include <memory>
#include <random>
#include <sstream>
#include <fstream>
#include <algorithm>
#include <fcntl.h>
#include <assert.h>
#include <string.h>
#include <stdlib.h>
#include <dirent.h>
#include <unistd.h>
#include <sys/stat.h>
#include <sys/mman.h>
#include <sys/types.h>
#include <immintrin.h>

#define ZMAP_PREDICT_FALSE(x) (__builtin_expect(!!(x), 0))
#define ZMAP_PREDICT_TRUE(x) (__builtin_expect(!!(x), 1))

#ifdef NDEBUG
#define zmap_perror(err)
#else
#define zmap_perror(err) fprintf(stderr, "%s: %s(errno: %d)\n", (err), \
        (0 == errno ? "Error in user codes" : strerror(errno)), errno), errno = 0
#endif

namespace zmap {

struct DefaultHasher {
    static inline uint64_t umul128(uint64_t a, uint64_t b, uint64_t* high) {
        __extension__ typedef unsigned __int128 __uint128;
        auto result = static_cast<__uint128>(a) * static_cast<__uint128>(b);
        *high = static_cast<uint64_t>(result >> 64);
        return static_cast<uint64_t>(result);
    }

    inline uint64_t operator()(uint64_t key) {
        // very fast mixing (similar to Abseil)
        static constexpr uint64_t k = 0xde5fb9d2630458e9ULL;
        uint64_t h = 0;
        uint64_t l = umul128(key, k, &h);
        return static_cast<size_t>(h + l);
    }
};

template<typename ... Args>
std::string string_printf(const std::string& fmt, Args ... args) {
    size_t size = std::snprintf(nullptr, 0, fmt.data(), args ...) + 1; // include '\0'
    auto buf = std::make_unique<char[]>(size);
    std::snprintf(buf.get(), size, fmt.data(), args ...);
    return std::string(buf.get(), size - 1); // remove '\0'
}
inline std::string string_printf(const std::string& fmt) { return fmt; }

inline size_t rand64() {
    static std::mt19937_64 mt((unsigned)time(nullptr));
    return mt();
}

inline bool string_ends_with(const std::string& str, const std::string& suffix) {
    if (suffix.size() > str.size()) return false;
    return std::equal(suffix.rbegin(), suffix.rend(), str.rbegin());
}

inline std::vector<std::string> list_files_under_dir(const std::string& path,
        const std::string& suffix = "", bool recursive = false) {
    std::vector<std::string> ret;
    if (path.empty()) return ret;

    const std::string& dir_path = (path.size() > 1 && '/' == path.back()) ?
            path.substr(0, path.size() - 1) : path;
    DIR* dir = opendir(dir_path.data());
    for(struct dirent entry, *p = &entry; dir && 0 == readdir_r(dir, &entry, &p) && p;) {
        if ('.' == *entry.d_name) continue;
        if (DT_DIR == entry.d_type) {
            if (!recursive) continue;
            auto&& sub_ret = list_files_under_dir(dir_path + "/" + entry.d_name, suffix, true);
            ret.insert(ret.end(), sub_ret.begin(), sub_ret.end());
        } else if (string_ends_with(entry.d_name, suffix)) {
            ret.emplace_back(dir_path + "/" + entry.d_name);
        }
    }
    closedir(dir);

    if (!dir && 0 == access(path.data(), F_OK)) ret.emplace_back(path);
    return ret;
}

constexpr static const uint8_t WRITABLE = 0;
constexpr static const uint8_t WRITE_LOCKED = -1;
#define MIN_TABLE_SIZE 10000

struct atomic_flag {
    inline bool test_and_set() {
        uint8_t writable_flag = WRITABLE;
        return __atomic_compare_exchange_n(&_flag, &writable_flag, WRITE_LOCKED,
                false, __ATOMIC_RELEASE, __ATOMIC_RELAXED);
    }
    inline void do_set() { while (!test_and_set()); }
    inline void clear() { __atomic_store_n(&_flag, WRITABLE, __ATOMIC_RELEASE); }
    inline explicit operator bool() const { return __atomic_load_n(&_flag, __ATOMIC_CONSUME); }
private:
    volatile uint8_t _flag = 0;
};

struct MmapUtility {
    constexpr static size_t min_size_for_malloc_with_mmap = 0x10000000; // 64MB
    static inline constexpr size_t round_up(size_t n) {
        constexpr size_t m = 0xFffFF; // 1MB
        return (n + m) & (~m);
    };

    static void* malloc(size_t size, bool not_dump = true, bool prior_use_hugepage = false) {
        void* addr = nullptr;
        auto alloc_size = size + 8; // reserve header space
        if (alloc_size < min_size_for_malloc_with_mmap) {
            // madvise requires memalign with pagesize
            if (not_dump) {
                if (0 != posix_memalign(&addr, 4096, alloc_size)) addr = nullptr;
            }
            if (!addr) addr = ::malloc(alloc_size);
        } else {
            alloc_size = round_up(alloc_size);
            addr = ::mmap(nullptr, alloc_size, PROT_READ | PROT_WRITE,
                      MAP_PRIVATE | MAP_ANONYMOUS | (prior_use_hugepage ? MAP_HUGETLB : 0), -1, 0);
            if (MAP_FAILED == addr && prior_use_hugepage) {
                addr = ::mmap(nullptr, alloc_size, PROT_READ | PROT_WRITE,
                            MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
            }
            if (MAP_FAILED == addr) { zmap_perror("mmap failed"); return nullptr; }
        }
        if (not_dump && madvise(addr, alloc_size, MADV_DONTDUMP) != 0) zmap_perror("madvise failed");
        *(size_t*)addr = alloc_size;
        return (size_t*)addr + 1;
    }

    static void free(void* p) {
        if (!p) return;
        p = (size_t*)p - 1;
        size_t size = *(size_t*)p;
        if (size < min_size_for_malloc_with_mmap) {
            ::free(p);
            return;
        }

        constexpr static size_t step = (1UL << 30); // 1GB
        while (size > step) {
            void* addr = mremap(p, size, size - step, 0);
            if (addr == MAP_FAILED) break;
            size -= step;
            if (addr != p) { p = addr; break; }
        }
        if (0 != ::munmap(p, size)) zmap_perror("munmap failed");
    }

    static void* mmap(int fd, size_t size, bool write_mode = false, bool not_dump = true,
            bool prior_use_hugepage = false) {
        if (fd < 0 || size == 0) return nullptr;
        auto mmap_helper = [](size_t length, int prot, int flags, int fd, off_t offset) {
            void* ret = ::mmap(nullptr, length, prot, flags, fd, offset);
            if (MAP_FAILED == ret && (flags & MAP_HUGETLB) && EINVAL == errno) {
                flags &= ~MAP_HUGETLB;
                return ::mmap(nullptr, length, prot, flags, fd, offset);
            }
            return ret;
        };

        void* addr = nullptr;
        if (write_mode) {
            addr = mmap_helper(size, PROT_READ | PROT_WRITE, (0 == getuid() ? MAP_LOCKED : 0) |
                    (prior_use_hugepage ? MAP_HUGETLB : 0) | MAP_SHARED | MAP_POPULATE, fd, 0);
        } else {
            addr = mmap_helper(size, PROT_READ, (0 == getuid() ? MAP_LOCKED : 0) |
                    (prior_use_hugepage ? MAP_HUGETLB : 0) | MAP_PRIVATE | MAP_POPULATE, fd, 0);
        }
        if (MAP_FAILED != addr && not_dump && madvise(addr, size, MADV_DONTDUMP) != 0)
            zmap_perror("madvise failed");

        return MAP_FAILED == addr ? nullptr : addr;
    }

    static void munmap(void* addr, size_t size, bool need_sync = false) {
        // msync's too slow, and there's no need to msync if munmap is called subsequently at once
        // if (need_sync && addr && 0 != msync(addr, size, MS_SYNC)) zmap_perror("msync failed");
        if (addr && 0 != ::munmap(addr, size)) zmap_perror("munmap failed");
    }
};

struct FileDataNode : public std::pair<void*, size_t> {
    enum MmapMode {
        FDNM_NO_MMAP = 0,
        FDNM_MMAP_READ = 1,
        FDNM_MMAP_WRITE = 2,
    };
    FileDataNode(void* addr, size_t sz, MmapMode mmap_mode = FDNM_NO_MMAP):
        std::pair<void*, size_t>(addr, sz), _mmap_mode(mmap_mode) {}
    FileDataNode(FileDataNode&& t): std::pair<void*, size_t>(t.first, t.second),
            _mmap_mode(t._mmap_mode) { t.first = nullptr; }
    FileDataNode& operator=(const FileDataNode&) = delete;
    ~FileDataNode() {
        FDNM_NO_MMAP == _mmap_mode ? MmapUtility::free(first) :
                MmapUtility::munmap(first, second, FDNM_MMAP_WRITE == _mmap_mode);
    }
private:
    MmapMode _mmap_mode = FDNM_NO_MMAP;
};

struct DumpWriter {
    virtual ~DumpWriter() {};
    virtual bool init_index(const std::string& index_file) = 0;
    virtual size_t write_index(const void* addr, size_t len) = 0;

    virtual bool init_data(const std::string& data_file) = 0;
    virtual size_t write_data(const void* addr, size_t len) = 0;
};

// we suggest you'd better avoid using mmap to load/dump, since it will degrade performance
// based our tests.
struct FileIOUtility {
    struct FileBaton {
        FileBaton(const std::string& path, bool write_mode, size_t size = 0, bool use_mmap = false):
            _write_mode(write_mode), _file_len(size) {
            _fd = open(path.data(), _write_mode ? (O_RDWR | O_CREAT) : O_RDONLY, 0666);
            if (-1 == _fd) { zmap_perror("open error"); return; }

            if (!_write_mode || 0 == _file_len) {
                struct stat st;
                if (-1 == fstat(_fd, &st)) zmap_perror("fstat error");
                else if (st.st_size > 0) _file_len = st.st_size;
            }
            if (_write_mode && use_mmap) {
                if (0 == _file_len) { zmap_perror("invalid file length"); return; }
                else if (-1 == ftruncate(_fd, _file_len)) zmap_perror("ftruncate error");
            }
            if (use_mmap) _mmap_addr = MmapUtility::mmap(_fd, _file_len, write_mode);
        }
        FileBaton(const FileBaton&) = delete;
        FileBaton& operator=(const FileBaton&) = delete;
        ~FileBaton() {
            MmapUtility::munmap(_mmap_addr, _file_len, _write_mode);
            if (-1 != _fd) close(_fd);
        }

        inline operator bool() { return -1 != _fd; }
        inline bool eof() const { return _off >= _file_len; }
        inline size_t size() const { return _file_len; }

        // if you wanna use raw pointer of mmap out of FileBaton, use can call
        // mmap_dismiss() after you've got the pointer via mmap_addr()
        inline void* mmap_addr() { return _mmap_addr; }
        inline void mmap_dismiss() { _mmap_addr = nullptr; }

        FileBaton& read(void* addr, size_t len) {
            if (_write_mode || _off + len > _file_len) return *this;
            char* data = (char*)addr;
            for (size_t i = 0, step = 1024 * 1024, n = std::min(step, len - i); i < len;
                    i += step, data += n, _off += n, n = std::min(step, len - i)) {
                if (_mmap_addr) memcpy(data, (char*)_mmap_addr + _off, n);
                else if ((ssize_t)n != ::read(_fd, data, n)) { zmap_perror("read error"); break; }
            }
            return *this;
        }

        FileBaton& write(const void* addr, size_t len) {
            if (!_write_mode || (_mmap_addr && _off + len > _file_len)) return *this;
            char* data = (char*)addr;
            for (size_t i = 0, step = 1024 * 1024, n = std::min(step, len - i); i < len;
                    i += step, data += n, _off += n, n = std::min(step, len - i)) {
                if (_mmap_addr) memcpy((char*)_mmap_addr + _off, data, n);
                else if ((ssize_t)n != ::write(_fd, data, n)) { zmap_perror("write error"); break; }
            }
            if (!_mmap_addr) _file_len = std::max(_file_len, _off);
            return *this;
        }

    private:
        int _fd = -1;
        bool _write_mode = false;
        size_t _file_len = 0;
        size_t _off = 0;
        void* _mmap_addr = nullptr;
    };

    // allocate memory for the whole content
    static FileDataNode load(const std::string& path, bool prior_load_via_mmap = true) {
        FileBaton baton(path, false, 0, prior_load_via_mmap);
        size_t size = baton.size();
        if (!baton || 0 == size) return {nullptr, 0};
        void* addr = MmapUtility::malloc(size);
        baton.read(addr, size);
        return {addr, size};
    }

    // multiple load, usage:
    // auto baton = mload(path);
    // while (baton && !baton.eof()) {
    //     baton.read(addr, len);
    // }
    static FileBaton mload(const std::string& path, bool use_mmap = false) {
        return {path, false, 0, use_mmap};
    }

    static bool dump(const std::string& path, void* data, size_t len,
            bool prior_dump_via_mmap = true) {
        FileBaton baton(path, true, len, prior_dump_via_mmap);
        return baton.write(data, len);
    }

    // multiple dump, usage:
    // auto baton = mdump(path);
    // while (baton) {
    //     baton.write(addr, len);
    // }
    static FileBaton mdump(const std::string& path, bool use_mmap = false, size_t len = -1) {
        return {path, true, len, use_mmap};
    }

    static FileDataNode mmap_whole_file(const std::string& path, bool write_mode) {
        FileBaton baton(path, write_mode, 0, true);
        auto addr = baton.mmap_addr();
        baton.mmap_dismiss();
        return {addr, baton.size(),
            write_mode ? FileDataNode::FDNM_MMAP_WRITE : FileDataNode::FDNM_MMAP_READ};
    }
};

template <typename KeyType>
struct KeyValueReader {
    virtual ~KeyValueReader() {}
    virtual bool next(KeyType& key, const void*& val, size_t* val_len, bool& erase_flag) = 0;

    // read should start from the beginning after clone
    virtual KeyValueReader* clone() = 0;
};

// key{delim}value: insert/update
// key: erase
// key{delim}: insert/update with empty value
// {delim}value: skip
template <typename KeyType>
struct KeyValueReaderFromFile : public KeyValueReader<KeyType> {
    ~KeyValueReaderFromFile() {
        free(_line);
        fclose(_stream);
    }
    void init(const std::string& file, char delim = '\t') {
        _file = file;
        _delim = delim;
        _stream = fopen(_file.data(), "r");
    }
    virtual bool next(KeyType& key, const void*& val, size_t* val_len, bool& erase_flag) {
        ssize_t n_read = 0;
        do {
            size_t line_len = 0;
            if (-1 == (n_read = getline(&_line, &line_len, _stream))) return false;
            if (n_read > 0 && '\n' == _line[n_read - 1]) {
                --n_read;
                _line[n_read] = '\0';
            }
        } while (n_read <= 0 || _delim == *_line);

        char* pend = strchr(_line, _delim);
        if (!parse_key(key, _line, pend ? pend - _line : n_read)) {
            zmap_perror(string_printf("parse key failed, line: %s", _line).data());
            return next(key, val, val_len, erase_flag);
        }
        val = nullptr;
        erase_flag = false;
        if (pend && _delim == *pend) {
            val = ++pend;
            if (val_len) *val_len = n_read - (pend - _line);
        } else {
            erase_flag = true;
        }
        return true;
    }
    virtual KeyValueReader<KeyType>* clone() {
        auto reader = new KeyValueReaderFromFile();
        reader->init(_file, _delim);
        return reader;
    }

private:
    template <typename T>
    static bool parse_key(T& key, const void* key_ptr, size_t key_len) {
        if (!key_ptr || 0 == key_len) return false;
        std::istringstream iss(std::string((const char*)key_ptr, key_len));
        iss >> key;
        return -1 == iss.tellg();
    }
    static bool parse_key(uint64_t& key, const void* key_ptr, size_t key_len) {
        if (!key_ptr || 0 == key_len) return false;
        char* key_end = nullptr;
        key = strtoul((const char*)key_ptr, &key_end, 10);
        return key_end && key_end == (const char*)key_ptr + key_len;
    }
    static bool parse_key(std::string& key, const void* key_ptr, size_t key_len) {
        if (!key_ptr || 0 == key_len) return false;
        key.assign((const char*)key_ptr, key_len);
        return true;
    }

private:
    std::string _file;
    char _delim = '\t';
    FILE* _stream = nullptr;
    char* _line = nullptr;
};

template <typename KeyType, typename FileReader = KeyValueReaderFromFile<KeyType>>
struct KeyValueReaderFromDir : public KeyValueReader<KeyType> {
    KeyValueReaderFromDir(const std::string& dir_path, char delim = '\t',
            const std::string& suffix = "") {
        for (const auto& file : list_files_under_dir(dir_path, suffix)) {
            auto reader = std::make_shared<FileReader>();
            reader->init(file, delim);
            _file_readers.push_back(reader);
        }
    }
    virtual bool next(KeyType& key, const void*& val, size_t* val_len, bool& erase_flag) {
        while (!_file_readers[_file_reader_idx]->next(key, val, val_len, erase_flag)) {
            if (++_file_reader_idx >= _file_readers.size()) {
                return false;
            }
        }
        return true;
    }
    virtual KeyValueReader<KeyType>* clone() {
        auto copy = new KeyValueReaderFromDir(*this);
        copy->_file_reader_idx = 0;
        for (auto& reader : copy->_file_readers) {
            reader.reset(reader->clone());
        }
        return copy;
    }

private:
    std::vector<std::shared_ptr<KeyValueReader<KeyType>>> _file_readers;
    size_t _file_reader_idx = 0;
};

constexpr static double MAX_LOAD_FACTOR = 7.0 / 8;

// return the real max_size in zmap based on the one that user expects
static inline uint64_t calc_max_size(uint64_t max_size) {
    uint64_t result = ~size_t{} >> __builtin_clzl(max_size);
    if (result * MAX_LOAD_FACTOR < max_size) {
        result <<= 1;
        result += 1;
    }
    return result * MAX_LOAD_FACTOR;
}

namespace internal {

inline std::string get_zmap_path(const std::string& path,
        const std::vector<std::string>& suffixes, bool allow_suffix_in_middle = false) {
    struct stat st;
    if (0 == stat(path.data(), &st) && S_ISDIR(st.st_mode)) {
        const auto& files = list_files_under_dir(path);
        for (const auto& suffix : suffixes) {
            for (const auto& file : files) {
                if (allow_suffix_in_middle) {
                    size_t pos = std::string::npos;
                    if (std::string::npos != (pos = file.find(suffix))) {
                        return file.substr(0, pos) + suffix;
                    }
                } else if (string_ends_with(file, suffix)) return file;
            }
        }
    } else {
        for (const auto& suffix : suffixes) {
            if (allow_suffix_in_middle) {
                size_t pos = std::string::npos;
                if (std::string::npos != (pos = path.find(suffix))) {
                    return path.substr(0, pos) + suffix;
                }
            } else if (string_ends_with(path, suffix)) return path;
        }
    }
    return path + (suffixes.empty() ? "" : suffixes.front());
}

// use murmurhash for string, which refers to
// https://sites.google.com/site/murmurhash/, and the source code
// is from https://sites.google.com/site/murmurhash/MurmurHash2_64.cpp
inline uint64_t MurmurHash64A(const void* key, int len, unsigned int seed = 0x5f5b463) {
    const uint64_t m = 0xc6a4a7935bd1e995;
    const int r = 47;

    uint64_t h = seed ^ (len * m);

    const uint64_t * data = (const uint64_t *)key;
    const uint64_t * end = data + (len/8);

    while(data != end) {
        uint64_t k = *data++;
        k *= m; k ^= k >> r; k *= m;
        h ^= k; h *= m;
    }

    const unsigned char* data2 = (const unsigned char*)data;
    switch(len & 7) {
        case 7: h ^= uint64_t(data2[6]) << 48;
        case 6: h ^= uint64_t(data2[5]) << 40;
        case 5: h ^= uint64_t(data2[4]) << 32;
        case 4: h ^= uint64_t(data2[3]) << 24;
        case 3: h ^= uint64_t(data2[2]) << 16;
        case 2: h ^= uint64_t(data2[1]) << 8;
        case 1: h ^= uint64_t(data2[0]); h *= m;
    };

    h ^= h >> r; h *= m; h ^= h >> r;
    return h;
}

using ctrl_t = char;
using h2_t = uint8_t;
using offset_t = size_t;
constexpr static size_t INVALID_OFFSET = -1;

enum Ctrl : ctrl_t {
    // kEmpty must be -128 if we want to use _mm_sign_epi8
    kEmpty    = -128, // 0b1000 0000
    kDeleted  = -127, // 0b1000 0001
    kSentinel = 0,    // 0b0000 0000
};

static ctrl_t h2_table[256];

#ifdef ZMAP_PROFILER
struct ZmapProfiler {
    ZmapProfiler() {
        for (auto& x : h2_hit_count_for_find) x = 0;
    }
    ~ZmapProfiler() { debug(); }

    void record_probe(size_t jump_cnt) {
        total_jump += jump_cnt;
        atomic_max(max_jump_in_one_probe, jump_cnt);
        ++probe_count;
        if (jump_cnt > 1) ++slowprobe_count;
    }

    void record_find(h2_t h2, uint64_t match_count, bool match) {
        ++h2_hit_count_for_find[h2];
        total_match_count += match_count;
        atomic_max(max_match_count_in_one_find, match_count);
        ++total_find;
        if (match) ++total_hit_find;
    }

    void debug(std::ostream* out = nullptr) {
        std::ostringstream oss;
        std::ostream* pout = out ? out : &oss;
        *pout << "\nzmap profiler:\n\n"
              << "insert related statistics:\n"
              << string_printf("\ttotal probe count: %lu\n", probe_count.load())
              << string_printf("\ttotal jump count: %lu\n", total_jump.load())
              << string_printf("\tmax jump count in one probe: %lu\n",
                      max_jump_in_one_probe.load())
              << string_printf("\tslow jump count: %lu\n", slowprobe_count.load())
              << string_printf("\tavg jump in one probe: %f\n",
                      probe_count ? total_jump * 1.0 / probe_count : 0.0)
              << "\n"
              << "find related statistics:\n"
              << string_printf("\ttotal find count: %lu\n", total_find.load())
              << string_printf("\ttotal hit count: %lu\n", total_hit_find.load())
              << string_printf("\thit rate: %f\n",
                      total_find ? total_hit_find * 1.0 / total_find : 0.0)
              << string_printf("\ttotal match count: %lu\n", total_match_count.load())
              << string_printf("\tmax match count in one find: %lu\n",
                      max_match_count_in_one_find.load())
              << string_printf("\tavg match count in one find: %f\n",
                      total_find ? total_match_count * 1.0 / total_find : 0.0)
              << "\n"
              << "h2 hit count:\n";
        for (size_t i = 0; i < h2_hit_count_for_find.size(); ++i) {
            *pout << string_printf("\th2[%lu]=%lu\n", i, h2_hit_count_for_find[i].load());
        }
        *pout << "\n";
        if (!out) printf("%s", oss.str().data());
    }

    static void atomic_max(std::atomic<uint64_t>& val, uint64_t new_val) {
        for (uint64_t n = val.load(std::memory_order_consume);
                new_val > n && val.compare_exchange_strong(n, new_val, std::memory_order_acq_rel);
                n = val.load(std::memory_order_consume));
    }

    // insert related statistics
    std::atomic<uint64_t> total_jump = {0};
    std::atomic<uint64_t> max_jump_in_one_probe = {0};
    std::atomic<uint64_t> probe_count = {0};
    std::atomic<uint64_t> slowprobe_count = {0};

    // find related statistics
    std::array<std::atomic<uint64_t>, 256> h2_hit_count_for_find;
    std::atomic<uint64_t> total_match_count = {0};
    std::atomic<uint64_t> max_match_count_in_one_find = {0};
    std::atomic<uint64_t> total_find = {0};
    std::atomic<uint64_t> total_hit_find = {0};
} g_profiler;
#endif

inline bool is_empty_or_deleted(ctrl_t c) { return c <= kDeleted; }

struct Header {
    uint32_t val_len;
    uint8_t group_width;
    uint8_t reserve1;
    uint8_t reserve2;
    uint8_t reserve3;
    size_t size;
    size_t deleted;
    size_t capacity;
};

template <size_t SIGNIFICANT_BITS>
struct BitMask {
    inline explicit BitMask(uint64_t n) : _n(n & _mask) {}
    inline BitMask& operator++() {
        _n &= (_n - 1);
        return *this;
    }
    inline explicit operator bool() const { return _n != 0; }
    inline BitMask operator&(BitMask other) const { return BitMask(_n | other._n); }
    inline int lowest_bit_set() const { return trailing_zeros(); }

    inline int trailing_zeros() const {
        return _n ? __builtin_ctzl(_n) : SIGNIFICANT_BITS;
    }

    inline int leading_zeros() const {
        return _n ? __builtin_clzl(_n << (64 - SIGNIFICANT_BITS)) : SIGNIFICANT_BITS;
    }

private:
    uint64_t _n;
    constexpr static uint64_t _mask = ((1UL << SIGNIFICANT_BITS) - 1);
};

// Groups without empty slots (but maybe with deleted slots) extend the probe
// sequence. The probing algorithm is quadratic. Given N the number of groups,
// the probing function for the i'th probe is:
//
//   P(0) = H1 % N
//
//   P(i) = (P(i - 1) + i) % N
//
// This probing function guarantees that after N probes, all the groups of the
// table will be probed exactly once.
class ProbeSeq {
public:
    inline ProbeSeq(size_t hash, size_t mask): _mask(mask), _offset(hash & mask) {
        assert(((mask + 1) & mask) == 0 && "invalid mask");
    }
#ifdef ZMAP_PROFILER
    ~ProbeSeq() { g_profiler.record_probe(_jump); }
#endif

    inline size_t offset() const { return _offset; }
    inline size_t offset(size_t i) const { return (_offset + i) & _mask; }
    inline void reset(size_t hash) {
        _offset = hash & _mask;
        _index = 0;
#ifdef ZMAP_PROFILER
        g_profiler.record_probe(_jump);
        _jump = 0;
#endif
    }

    inline void next(size_t group_width) {
        _index += group_width;
        _offset += _index;
        _offset &= _mask;
#ifdef ZMAP_PROFILER
        ++_jump;
#endif
    }

private:
    size_t _mask = 0;
    size_t _offset = 0;
    size_t _index = 0;
#ifdef ZMAP_PROFILER
    size_t _jump = 0;
#endif
};

// We don't want to define a base class like ProbeGroup using polymorphism,
// since it will degrade performance dramatically.
#define PROBE_GROUP_COMMON_PARTS(Type)                                 \
    enum { kWidth = sizeof(Type) };                                    \
    inline auto match(h2_t hash) const {                               \
        return BitMask<kWidth>(match_internal(gen_type(hash), _ctrl)); \
    }                                                                  \
    inline int match_first_empty() const {                             \
        static Type s_empty = gen_type(kEmpty);                        \
        auto r = match_internal(s_empty, _ctrl);                       \
        return r ? __builtin_ctzl(r) : -1;                             \
    }                                                                  \
    inline int match_first_empty_or_deleted() const {                  \
        auto r = match_empty_or_deleted();                             \
        return r ? __builtin_ctzl(r) : -1;                             \
    }                                                                  \
    inline auto match_empty() const {                                  \
        static Type s_empty = gen_type(kEmpty);                        \
        return BitMask<kWidth>(match_internal(s_empty, _ctrl));        \
    }                                                                  \
    inline uint32_t count_leading_empty_or_deleted() const {           \
        BitMask<kWidth> mask(match_empty_or_deleted() + 1);            \
        return mask ? mask.trailing_zeros() : kWidth;                  \
    }                                                                  \
private:                                                               \
    Type _ctrl

struct ProbeGroup128 {
    inline void reset(const ctrl_t* pos) {
        _ctrl = _mm_loadu_si128(reinterpret_cast<const __m128i*>(pos));
    }

    static inline __m128i gen_type(h2_t hash) { return _mm_set1_epi8(hash); }

    inline int match_empty_or_deleted() const {
        static __m128i s_threshold = gen_type(kDeleted + 1);
        return _mm_movemask_epi8(_mm_cmplt_epi8(_ctrl, s_threshold));
    }

    static inline int match_internal(__m128i l, __m128i r) {
        return _mm_movemask_epi8(_mm_cmpeq_epi8(l, r));
    }

    PROBE_GROUP_COMMON_PARTS(__m128i);
};

// compile with -mavx2
#if defined(__AVX__) && defined(__AVX2__)
struct ProbeGroup256 {
    inline void reset(const ctrl_t* pos) {
        _ctrl = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(pos));
    }

    static inline __m256i gen_type(h2_t hash) { return _mm256_set1_epi8(hash); }

    inline int match_empty_or_deleted() const {
        static __m256i s_threshold = gen_type(kDeleted + 1);
        return _mm256_movemask_epi8(_mm256_cmpgt_epi8(s_threshold, _ctrl));
    }

    static inline int match_internal(__m256i l, __m256i r) {
        return _mm256_movemask_epi8(_mm256_cmpeq_epi8(l, r));
    }

    PROBE_GROUP_COMMON_PARTS(__m256i);
};
#endif

// compile with -march=skylake-avx512
#if defined(__AVX512F__) && defined(__AVX512BW__) && __GNUC__ >= 6
struct ProbeGroup512 {
    inline void reset(const ctrl_t* pos) {
        _ctrl = _mm512_loadu_si512(reinterpret_cast<const __m512i*>(pos));
    }

    static inline __m512i gen_type(h2_t hash) { return _mm512_set1_epi8(hash); }

    inline int match_empty_or_deleted() const {
        static __m512i s_threshold = gen_type(kDeleted + 1);
        return _mm512_cmplt_epi8_mask(_ctrl, s_threshold));
    }

    static inline uint64_t match_internal(__m512i l, __m512i r) {
        return _mm512_cmpeq_epi8_mask(l, r);
    }

    PROBE_GROUP_COMMON_PARTS(__m512i);
};
#endif

#undef PROBE_GROUP_COMMON_PARTS

template <typename, typename, typename> class ElasticZmapImpl;
template <typename, typename> class StringZmapImpl;
template <typename, size_t> struct ParallelZmapImpl;

// ----------------------------------------------------------------------------
//                     Z M A P
// ----------------------------------------------------------------------------
// An open-addressing hashtable with quadratic probing.
//
// ABOUT THE NAME
// 'Z' is first alpha of my surname 'Zhao', and meanwhile 'Z' is the last alpha
// in alphabet, expecting it's the final solution for everyone.
//
// IMPLEMENTATION DETAILS
//
// The table stores elements inline in a slot array. In addition to the slot
// array the table maintains some control state per slot. The extra state is one
// byte per slot and stores empty or deleted marks, or alternatively 8 bits from
// the hash of an occupied slot. The table is split into logical groups of
// slots, like so:
//
//      Group 1         Group 2        Group 3
// +---------------+---------------+---------------+
// | | | ... | | | | | | ... | | | | | | ... | | | |
// +---------------+---------------+---------------+
//
// On lookup the hash is split into two parts:
// - H2: control part
// - H1: index part
// The groups are probed using H1. For each group the slots are matched to H2 in
// parallel. Because H2 is 8 bits (factually 253 states) and the number of slots
// per group is low (16 or 32/64), in almost all cases a match in H2 is also
// a lookup hit.
//
// On insert, once the right group is found (as in lookup), its slots are
// filled in order.
//
// On erase a slot is cleared. In case the group did not have any empty slots
// before the erase, the erased slot is marked as deleted (or empty under some
// conditions).
template <typename ProbeGroupType, size_t ValueLength, typename Hasher = DefaultHasher>
class ZmapImpl {
    template <bool IS_WRITE_MODE>
    struct ScopedLock {
        ScopedLock(uint8_t* d): _data(d) {
            if (!_data) return;
            if (IS_WRITE_MODE) {
                for (uint8_t n = WRITABLE; !__atomic_compare_exchange_n(_data, &n,
                        WRITE_LOCKED, false, __ATOMIC_RELEASE, __ATOMIC_RELAXED); n = WRITABLE);
            } else {
                for (uint8_t n = __atomic_load_n(_data, __ATOMIC_CONSUME);
                        !(n < WRITE_LOCKED - 1 && __atomic_compare_exchange_n(_data, &n, n + 1,
                                false, __ATOMIC_RELEASE, __ATOMIC_RELAXED));
                        n = __atomic_load_n(_data, __ATOMIC_CONSUME));
            }
        }
        ScopedLock(const ScopedLock&) = delete;
        ScopedLock& operator=(const ScopedLock&) = delete;
        ~ScopedLock() {
            if (!_data) return;
            if (IS_WRITE_MODE) {
                assert(WRITE_LOCKED == *_data);
                __atomic_store_n(_data, WRITABLE, __ATOMIC_RELEASE);
            } else {
                assert(WRITE_LOCKED != *_data && WRITABLE != *_data);
                __atomic_sub_fetch(_data, 1, __ATOMIC_ACQ_REL);
            }
        }
    private:
        volatile uint8_t* _data;
    };
    using ScopedReadLock = ScopedLock<false>;
    using ScopedWriteLock = ScopedLock<true>;

    struct __attribute__((packed, aligned(1))) Slot {
        union {
            uint64_t key;
            uint64_t first; // alias for key
        };
        union {
            uint8_t val[ValueLength];
            uint8_t second[ValueLength]; // alias for value
        };
    };

public:
    typedef uint64_t KeyType;
    typedef typename std::remove_cv<typename std::remove_reference<KeyType>::type>::type RawKeyType;
    typedef Hasher HasherType;
    ZmapImpl() { memset(&_header, 0, sizeof(_header)); }

    // build a new zmap
    ZmapImpl(uint64_t max_size) {
        bool init_result = init(max_size);
        assert(init_result && "init failed");
        (void)init_result;
    }

    // load an existed zmap
    ZmapImpl(const std::string& path, bool use_mmap = false) {
        bool load_result = load(path, use_mmap);
        assert(load_result && "load failed");
        (void)load_result;
    }

    bool init(uint64_t max_size, size_t sub_idx = -1) {
        if (NULL != _ctrl) {
            zmap_perror("already init ok");
            return false;
        }

        { // It's ok even if init by multiple threads
            for (int i = 0; i < 256; ++i) {
                h2_table[i] = i;
            }
            h2_table[(uint8_t)kEmpty] = (ctrl_t)251;
            h2_table[(uint8_t)kDeleted] = (ctrl_t)241;
            h2_table[(uint8_t)kSentinel] = (ctrl_t)239;
        }

        memset(&_header, 0, sizeof(_header));

        if (max_size < MIN_TABLE_SIZE) max_size = MIN_TABLE_SIZE;

        // rounds up to the next power of 2 minus 1
        _header.capacity = ~size_t{} >> __builtin_clzl(max_size);
        if (_header.capacity * MAX_LOAD_FACTOR < max_size) {
            _header.capacity <<= 1;
            _header.capacity += 1;
        }

        _header.group_width = ProbeGroupType::kWidth;
        _header.val_len = ValueLength;
        _max_size = _header.capacity * MAX_LOAD_FACTOR;

        void* addr = MmapUtility::malloc(sizeof(Header) + _header.capacity +
                ProbeGroupType::kWidth + 1 + sizeof(Slot) * _header.capacity);
        if (NULL == addr) {
            zmap_perror("malloc error");
            return false;
        }
        _allocated_datas.emplace_back(addr, [](void* addr) { MmapUtility::free(addr); });

        _ctrl = (ctrl_t*)addr + sizeof(Header);
        memset(_ctrl, kEmpty, _header.capacity + ProbeGroupType::kWidth + 1);
        _ctrl[_header.capacity] = kSentinel;
        _ctrl_end = _ctrl + _header.capacity;

        _mtxs = (uint8_t*)MmapUtility::malloc(_header.capacity);
        if (NULL == _mtxs) {
            zmap_perror("malloc error");
            return false;
        }
        _allocated_datas.emplace_back(_mtxs, [](void* addr) { MmapUtility::free(addr); });
        memset(_mtxs, WRITABLE, _header.capacity);

        _slots = reinterpret_cast<Slot*>(_ctrl_end + ProbeGroupType::kWidth + 1);
        return true;
    }

    // If use_mmap is true, the disk file will be consistent with modifications in memory and
    // there's no need to dump at the end.
    // What's more, it's not necessary that path_raw ends with a suffix like ".zmap_*".
    bool load(const std::string& path_raw, bool use_mmap = false) {
        if (NULL != _ctrl) {
            zmap_perror("already load ok");
            return false;
        }

        { // It's ok even if init by multiple threads
            for (int i = 0; i < 256; ++i) {
                h2_table[i] = i;
            }
            h2_table[(uint8_t)kEmpty] = (ctrl_t)251;
            h2_table[(uint8_t)kDeleted] = (ctrl_t)241;
            h2_table[(uint8_t)kSentinel] = (ctrl_t)239;
        }

        memset(&_header, 0, sizeof(_header));

        const std::string& path = get_zmap_path(path_raw, {".zmap_one", ".zmap_idx"});
        if (use_mmap) {
            _loaded_data = std::make_unique<FileDataNode>(FileIOUtility::mmap_whole_file(path, true));
            _mmap_file = path;
        } else {
            _loaded_data = std::make_unique<FileDataNode>(FileIOUtility::load(path, false));
        }
        auto addr = (uint8_t*)_loaded_data->first;
        if (!addr) return false;

        Header* header = (Header*)addr;
        memcpy(&_header, header, sizeof(Header));
        if (ProbeGroupType::kWidth != _header.group_width) {
            zmap_perror("group width inconsistent");
            return false;
        }
        if (ValueLength != _header.val_len) {
            zmap_perror("value length inconsistent");
            return false;
        }
        _max_size = _header.capacity * MAX_LOAD_FACTOR;

        _ctrl = (ctrl_t*)(addr + sizeof(Header));
        _ctrl_end = _ctrl + _header.capacity;

        _mtxs = (uint8_t*)MmapUtility::malloc(_header.capacity);
        if (NULL == _mtxs) {
            zmap_perror("malloc error");
            return false;
        }
        _allocated_datas.emplace_back(_mtxs, [](void* addr) { MmapUtility::free(addr); });
        memset(_mtxs, WRITABLE, _header.capacity);

        _slots = reinterpret_cast<Slot*>(_ctrl_end + ProbeGroupType::kWidth + 1);
        return true;
    }

    // the filename saved finally will add a suffix with ".zmap_one"(including value part)
    // or ".zmap_idx"(just for index part), and ".part-XXX" will also be added as the part
    // of suffix for parallel zmap
    bool dump(const std::string& path_raw, size_t sub_idx = -1, DumpWriter* writer = nullptr) {
        const std::string& path = get_zmap_path(path_raw, {".zmap_one", ".zmap_idx"});
        if (path == _mmap_file) return true;

        size_t sz = sizeof(Header) + _header.capacity + ProbeGroupType::kWidth + 1
                + sizeof(Slot) * _header.capacity;

        if (writer) {
            if (!writer->init_index(path)) return false;
            return sizeof(Header) == writer->write_index(&_header, sizeof(Header)) &&
                    sz - sizeof(Header) == writer->write_index(_ctrl, sz - sizeof(Header));
        }

        auto&& baton = FileIOUtility::mdump(path);
        if (!baton) return false;
        return baton.write(&_header, sizeof(Header)) && baton.write(_ctrl, sz - sizeof(Header));
    }

    // remove data from disk
    bool unlink(const std::string& path_raw) {
        return 0 == ::unlink(get_zmap_path(path_raw, {".zmap_one", ".zmap_idx"}).data());
    }
    inline static size_t subcnt() { return 1UL; }
    inline static size_t subidx(size_t hash_val) { return 0UL; }

    struct const_iterator {
        const_iterator() = default;
        const_iterator& operator=(const const_iterator& other) = default;
        inline const Slot& operator*() const {
            if (_slot_inited_flag.test_and_set()) {
                if (_ctrl < _ctrl_end) {
                    ScopedReadLock lock(_mtx);
                    memcpy(&_slot, _slots, sizeof(Slot));
                } else {
                    // considering ElasticZmap's value is an offset, so using -1 to init is better
                    memset(&_slot, 0xFF, sizeof(Slot));
                }
            }
            return _slot;
        }
        inline const Slot* operator->() const { return &operator*(); }

        inline const_iterator& operator++() {
            if (_ctrl < _ctrl_end) {
                ++_ctrl;
                ++_mtx;
                ++_slots;
                skip_empty_or_deleted();
                _slot_inited_flag.clear();
            }
            return *this;
        }

        inline friend bool operator==(const const_iterator& a, const const_iterator& b) {
            return a._ctrl == b._ctrl;
        }
        inline friend bool operator!=(const const_iterator& a, const const_iterator& b) {
            return !(a == b);
        }
    private:
        template <typename, size_t, typename> friend class ZmapImpl;
        friend struct DefaultAlloc;
        template <typename, typename, typename> friend class ElasticZmapImpl;
        template <typename, size_t> friend struct ParallelZmapImpl;
        const_iterator(ctrl_t* ctrl, uint8_t* mtx, Slot* slots, ctrl_t* ctrl_end):
            _ctrl(ctrl), _mtx(mtx), _slots(slots), _ctrl_end(ctrl_end) {}

        inline void skip_empty_or_deleted() {
            ProbeGroupType group;
            while (is_empty_or_deleted(*_ctrl)) {
                group.reset(_ctrl);
                auto shift = group.count_leading_empty_or_deleted();
                _ctrl += shift;
                _mtx += shift;
                _slots += shift;
            }
        }

        // use the feature that unique_ptr can't copy construction
        template <bool IS_WRITE_MODE>
        std::unique_ptr<ScopedLock<IS_WRITE_MODE>> lock() const {
            return std::make_unique<ScopedLock<IS_WRITE_MODE>>(_mtx);
        }
        Slot* raw_slot() const {
            return (_ctrl < _ctrl_end && !is_empty_or_deleted(*_ctrl)) ?
                    const_cast<Slot*>(_slots) : nullptr;
        }

        ctrl_t* _ctrl = nullptr;
        uint8_t* _mtx = nullptr;
        Slot* _slots = nullptr;
        ctrl_t* _ctrl_end = nullptr;
        mutable Slot _slot;
        mutable atomic_flag _slot_inited_flag;
    };

    inline const_iterator begin() const {
        if (empty()) return end();
        const_iterator it{_ctrl, _mtxs, _slots, _ctrl_end};
        it.skip_empty_or_deleted();
        return it;
    }
    inline const_iterator end() const { return {_ctrl_end, nullptr, nullptr, _ctrl_end}; }

    // we should keep the uniform emplace interface for Zmap/ElasticZmap/ParallelZmap
    inline std::pair<const_iterator, bool> emplace(uint64_t key, const void* val,
            size_t val_len, size_t hash_val = -1) {
        bool insert_ok = false;
        size_t index = insert_internal(key, val, insert_ok, hash_val);
        if (ZMAP_PREDICT_FALSE((size_t)-1 == index)) return {end(), false};
        return {{_ctrl + index, _mtxs + index, _slots + index, _ctrl_end}, insert_ok};
    }
    inline std::pair<const_iterator, bool> emplace_s(uint64_t key, const std::string& val,
            size_t hash_val = -1) {
        return emplace(key, (const void*)val.data(), val.size(), hash_val);
    }

    // return -1 if emplace failed, e.g. map is full
    // return 0 if key already exists
    // return 1 if insert ok
    inline int emplace_ex(uint64_t key, const void* val, size_t val_len = 0,
            size_t hash_val = -1) {
        bool insert_ok = false;
        auto index = insert_internal(key, val, insert_ok, hash_val);
        if ((size_t)-1 == index) return -1;
        return insert_ok ? 1 : 0;
    }

    inline std::pair<const_iterator, bool> insert(const std::pair<uint64_t, const void*>& kv,
            size_t hash_val = -1) {
        bool insert_ok = false;
        size_t index = insert_internal(kv.first, kv.second, insert_ok, hash_val);
        if (ZMAP_PREDICT_FALSE((size_t)-1 == index)) return {end(), false};
        return {{_ctrl + index, _mtxs + index, _slots + index, _ctrl_end}, insert_ok};
    }
    inline std::pair<const_iterator, bool> insert(const std::pair<uint64_t, std::string>& kv,
            size_t hash_val = -1) {
        return insert({kv.first, (const void*)kv.second.data()}, hash_val);
    }

    // ${return_val}.second is true if key doesn't exist before
    inline std::pair<const_iterator, bool> update(uint64_t key, const void* val, size_t val_len,
            size_t hash_val = -1) {
        bool insert_ok = false;
        size_t index = insert_internal<false, true>(key, val, insert_ok, hash_val);
        if (ZMAP_PREDICT_FALSE((size_t)-1 == index)) return {end(), false};
        return {{_ctrl + index, _mtxs + index, _slots + index, _ctrl_end}, insert_ok};
    }
    inline std::pair<const_iterator, bool> update_s(uint64_t key, const std::string& val,
            size_t hash_val = -1) {
        return update(key, val.data(), val.size(), hash_val);
    }
    // the meaning of return value is the same as the one of emplace_ex
    inline int update_ex(uint64_t key, const void* val, size_t val_len, size_t hash_val = -1) {
        bool insert_ok = false;
        size_t index = insert_internal<false, true>(key, val, insert_ok, hash_val);
        return (size_t)-1 == index ? -1 : (int)insert_ok;
    }

    // build zmap from raw data with two extra features:
    // 1. there's no key that is deleted until build finishes;
    // 2. use the value occurring firstly for duplicate key;
    // what's more, please set the template parameter 'true' if there's no
    // duplicate key in your dataset.
    // the meaning of return value is same as the one of emplace_ex
    template <bool NO_DUP_KEY = false>
    inline int build(uint64_t key, const void* val, size_t val_len = 0, size_t hash_val = -1) {
        bool insert_ok = false;
        auto index = insert_internal<NO_DUP_KEY, false, true>(key, val, insert_ok, hash_val);
        if ((size_t)-1 == index) return -1;
        return insert_ok ? 1 : 0;
    }

    inline size_t erase(uint64_t key, size_t hash_val = -1) {
        const size_t index = find_internal(key, hash_val);
        return erase_internal(index);
    }

    inline size_t erase(const const_iterator& it) {
        if (end() == it) return 0;
        return erase_internal(it._slots - _slots);
    }

    inline const_iterator find(uint64_t key, size_t hash_val = -1) const {
        size_t index = find_internal(key, hash_val);
        if ((size_t)-1 == index) return end();
        return {_ctrl + index, _mtxs + index, _slots + index, _ctrl_end};
    }

    // the implementation of const_iterator will construct a string object via copy value,
    // so find_ex could provide the pointer of value directly, but you should take the risk
    // that the content of pointer might be modified and you'd better just read it without
    // any modification.
    inline bool find_ex(uint64_t key, const void*& val, size_t* val_len = nullptr,
            size_t hash_val = -1) const {
        size_t index = find_internal(key, hash_val);
        if ((size_t)-1 == index) return false;
        val = (_slots + index)->val;
        if (val_len) *val_len = ValueLength;
        return true;
    }

    inline size_t count(uint64_t key, size_t hash_val = -1) const {
        return (size_t)-1 != find_internal(key, hash_val);
    }

    // using the raw pointer should take the risk that the content of pointer might be modified
    inline const void* at(uint64_t key, size_t hash_val = -1) const {
        size_t index = find_internal(key, hash_val);
        if ((size_t)-1 == index) abort();
        return (_slots + index)->val;
    }

    // using the raw pointer should take the risk that the content of pointer might be modified
    inline const void* operator[](uint64_t key) {
        bool insert_ok = false;
        size_t index = insert_internal<false, false>(key, nullptr, insert_ok);
        return (size_t)-1 == index ? nullptr : (_slots + index)->val;
    }

    // just facilitate for test
    inline uint64_t rand_key() const {
        if (empty()) return -1;
        size_t idx = rand64() % _header.capacity;
        const_iterator it{_ctrl + idx, _mtxs + idx, _slots + idx, _ctrl_end};
        it.skip_empty_or_deleted();
        if (end() != it) return it->first;
        return begin()->first;
    }

    inline size_t size() const { return _header.size; }
    inline bool empty() const { return 0 == size(); }
    // for user, we should return _max_size as capacity, and _header.capacity is
    // an internal parameter in zmap's implementation
    inline size_t capacity() const { return _max_size; }

    inline void clear() {
        if (!_ctrl) return;
        memset(_ctrl, kEmpty, _header.capacity + ProbeGroupType::kWidth + 1);
        _ctrl[_header.capacity] = kSentinel;
        _header.size = 0;
        _header.deleted = 0;
        _header.reserve1 = 0;
        _header.reserve2 = 0;
        _header.reserve3 = 0;
    }
    inline void reset() {
        *this = {};
    }

    void debug(std::ostream* out = nullptr, size_t sub_idx = -1) const {
        std::ostringstream oss;
        std::ostream* pout = out ? out : &oss;
        *pout << string_printf("[zmap] size: %lu, capacity: %lu, max load factor: %f, "
                "deleted: %lu, max_size: %lu, left: %lu\n", _header.size, _header.capacity,
                MAX_LOAD_FACTOR, _header.deleted, _max_size, _max_size - _header.size);
        if (!out) printf("%s", oss.str().data());
    }

private:
    size_t find_internal(uint64_t key, size_t hash_val = -1) const {
        if ((size_t)-1 == hash_val) hash_val = Hasher()(key);
        h2_t h2 = H2(hash_val);
        ProbeSeq seq(H1(hash_val), _header.capacity);
        ProbeGroupType group;
#ifdef ZMAP_PROFILER
        uint64_t match_count = 0;
#endif
        while (true) {
            group.reset(_ctrl + seq.offset());
            for (auto&& match = group.match(h2); match; ++match) {
                size_t index = seq.offset(match.lowest_bit_set());
                Slot* item = _slots + index;
#ifdef ZMAP_PROFILER
                ++match_count;
#endif
                if (ZMAP_PREDICT_TRUE(key == item->key)) {
#ifdef ZMAP_PROFILER
                    g_profiler.record_find(h2, match_count, true);
#endif
                    return index;
                }
            }
            if (ZMAP_PREDICT_TRUE(group.match_first_empty() >= 0)) {
#ifdef ZMAP_PROFILER
                g_profiler.record_find(h2, match_count, false);
#endif
                return -1;
            }
            seq.next(ProbeGroupType::kWidth);
        }

#ifdef ZMAP_PROFILER
        g_profiler.record_find(h2, match_count, false);
#endif
        return -1;
    }

    template <bool SKIP_CHECK_EXISTENCE = false, bool OVERWRITE = false, bool BUILD_MODE = false>
    size_t insert_internal(uint64_t key, const void* val, bool& insert_ok, size_t hash_val = -1) {
        // considering ElasticZmap's value is an offset, so using -1 to init is better
        static uint8_t s_empty_value[ValueLength];
        static atomic_flag s_init_empty_value;
        if (s_init_empty_value.test_and_set()) {
            memset(s_empty_value, 0xFF, ValueLength);
        }
        if (!val) val = s_empty_value;
        if ((size_t)-1 == hash_val) hash_val = Hasher()(key);
        size_t h1 = H1(hash_val);
        h2_t h2 = H2(hash_val);
        ProbeSeq seq(h1, _header.capacity);

        size_t insert_pos = -1;

        // group must be allocated on stack with regard to alignment of __mXXXi,
        // refer to https://stackoverflow.com/questions/55778692 for details
        ProbeGroupType group;
        if (!SKIP_CHECK_EXISTENCE) {
            while (true) {
                group.reset(_ctrl + seq.offset());
                for (auto&& match = group.match(h2); match; ++match) {
                    size_t index = seq.offset(match.lowest_bit_set());
                    Slot* item = _slots + index;
                    if (ZMAP_PREDICT_TRUE(key == item->key)) {
                        if (OVERWRITE) {
                            if (BUILD_MODE) { // no need to lock for build mode
                                memcpy(item->val, val, ValueLength);
                            } else {
                                ScopedWriteLock lock(_mtxs + index);
                                memcpy(item->val, val, ValueLength);
                            }
                        }
                        return index;
                    }
                }

                // if _header.size + _header.deleted == _header.capacity, it'll dead loop
                assert(_header.size + _header.deleted < _header.capacity
                        && "too many deleted keys");
                auto r = group.match_first_empty();
                if (ZMAP_PREDICT_TRUE(r >= 0)) {
                    insert_pos = seq.offset(r);
                    break;
                }
                seq.next(ProbeGroupType::kWidth);
            }
        }

        if (ZMAP_PREDICT_FALSE(_header.size >= _max_size)) return -1;

        // no deleted items for build mode
        if (SKIP_CHECK_EXISTENCE || !BUILD_MODE) {
            seq.reset(h1);
            while (true) {
                group.reset(_ctrl + seq.offset());

                auto r = group.match_first_empty_or_deleted();
                if (ZMAP_PREDICT_TRUE(r >= 0)) {
                    insert_pos = seq.offset(r);
                    break;
                }
                seq.next(ProbeGroupType::kWidth);
            }
            if (kDeleted == _ctrl[insert_pos]) --_header.deleted;
        }

        ++_header.size;
        set_ctrl(insert_pos, h2);

        Slot* item = _slots + insert_pos;
        if (BUILD_MODE) { // no need to lock for build mode
            item->key = key;
            memcpy(item->val, val, ValueLength);
        } else {
            ScopedWriteLock lock(_mtxs + insert_pos);
            item->key = key;
            memcpy(item->val, val, ValueLength);
        }

        insert_ok = true;
        return insert_pos;
    }

    size_t erase_internal(size_t index) {
        if ((size_t)-1 == index) return 0;

        ProbeGroupType group;
        --_header.size;
        const size_t index_before = (index - ProbeGroupType::kWidth) & _header.capacity;
        group.reset(_ctrl + index);
        const auto empty_after = group.match_empty();
        group.reset(_ctrl + index_before);
        const auto empty_before = group.match_empty();

        // We count how many consecutive non empties we have to the right and to the
        // left of `it`. If the sum is >= kWidth then there is at least one probe
        // window that might have seen a full group.
        bool was_never_full =
            empty_before && empty_after &&
            static_cast<size_t>(empty_after.trailing_zeros() +
                empty_before.leading_zeros()) < ProbeGroupType::kWidth;

        _header.deleted += !was_never_full;
        set_ctrl(index, was_never_full ? kEmpty : kDeleted);
        return 1;
    }

    template <typename, typename, typename> friend class ElasticZmapImpl;
    template <typename, size_t> friend struct ParallelZmapImpl;
    inline auto& header() { return _header; }
    inline const auto& header() const { return _header; }

private:
    static inline size_t H1(size_t hash) { return (hash >> 7); }
    static inline ctrl_t H2(size_t hash) { return h2_table[hash & 0xFF]; }

private:
    // Sets the control byte, and if `i < _group_width`, set the cloned byte at
    // the end too.
    inline void set_ctrl(size_t i, ctrl_t h) {
        assert(i < _header.capacity);

        _ctrl[i] = h;
        if (i < ProbeGroupType::kWidth) {
            _ctrl[((i - ProbeGroupType::kWidth) & _header.capacity) + 1 +
              ((ProbeGroupType::kWidth - 1) & _header.capacity)] = h;
        }
    }

private:
    Header _header;
    ctrl_t* _ctrl = nullptr;
    ctrl_t* _ctrl_end = nullptr;
    uint8_t* _mtxs = nullptr;
    Slot* _slots = nullptr;
    size_t _max_size = 0;
    std::string _mmap_file;
    std::vector<std::shared_ptr<void>> _allocated_datas;
    std::unique_ptr<FileDataNode> _loaded_data;
};

// ----------------------------------------------------------------------------
//                     A L L O C A T O R
// ----------------------------------------------------------------------------
// An allocator supports load/dump with disk.
//
// item structure:
//   4-byte length header(exclude the length of 4-byte header) + data
// 4-byte length header structure:
//   1-bit free_list_flag + 4-bit rear padding + 27-bit data length
// item structure of normal item:
//   4-byte length header(with '0' free_list_flag) + data
// item structure of item in free list:
//   4-byte length header(with '1' free_list_flag) + 8-byte next offset in free list + other data
//
// constraints:
//   1. just one thread to write;
//   2. data length should be less than 2^27 - 4 (128M or so)
struct DefaultAlloc {
    using len_head_t = uint32_t;
    struct __attribute__((packed, aligned(1))) FreeNode {
        len_head_t len; // length for current free node
        offset_t next_off; // offset for next free node
    };
    struct MemBlock {
        uint8_t* block = nullptr;
        size_t off = 0;
    };

    ~DefaultAlloc() { clear(); }

    void clear() {
        _stop_defrag = true;
        while (_doing_defrag) usleep(1);
        { // make sure defrag thread has done
            std::lock_guard<std::mutex> lock(_interim_free_nodes_mtx);
        }
        _stop_defrag = false;

        for (const auto& mem_block : _mem_blocks) {
            MmapUtility::free(mem_block.block);
        }
        _mem_blocks.clear();
        _free_lists.clear();
        _interim_free_nodes.clear();
        _head_of_off = nullptr;
        _sorted_free_sizes.clear();
        _free_node_cnt = 0;
        _last_free_node_cnt = 0;
    }
    void reset() {
        clear();
        _free_lists.reset();
    }

    // seek available offset
    offset_t try_alloc(size_t len) {
        if (_free_node_cnt > 1024 * 1024 && _free_node_cnt > _last_free_node_cnt * 1.3) {
            defrag();
        }

        if (0 == len || len > MAX_DATA_LENGTH) return INVALID_OFFSET;
        len = sizeof(len_head_t) + (len > MIN_DATA_LENGTH ? len : MIN_DATA_LENGTH);

        offset_t off = INVALID_OFFSET;
        // try to alloc from free lists firstly
        if (!_doing_defrag) {
            auto it = _sorted_free_sizes.lower_bound(len);
            if (_sorted_free_sizes.end() != it) {
                size_t free_len = *it;
                const void* free_len_val = nullptr;
                if (_free_lists.find_ex(free_len, free_len_val)) {
                    _head_of_off = (offset_t*)free_len_val;
                    off = *_head_of_off;
                    assert(std::max((uint32_t)MIN_DATA_LENGTH, ((FreeNode*)addr(off))->len &
                            (uint32_t)DATA_LENGTH_MASK) + sizeof(len_head_t) == free_len &&
                            "length in free node abnormal");
                }
            }
        }

        if (INVALID_OFFSET == off) {
            size_t mem_block_count = _mem_blocks.size();
            if (0 == mem_block_count) {
                off = 0;
            } else {
                auto cur_mem_block_off = _mem_blocks.back().off;
                if (cur_mem_block_off + len > BLOCK_SIZE || BLOCK_SIZE -
                        cur_mem_block_off < MIN_ALLOC_LENGTH) { // create a new block
                    off = mem_block_count * BLOCK_SIZE;
                } else {
                    off = (mem_block_count - 1) * BLOCK_SIZE + cur_mem_block_off;
                }
            }
        }

        return off;
    }
    void* alloc(offset_t off, size_t len) {
        if (len > MAX_DATA_LENGTH || 0 == len || INVALID_OFFSET == off) return nullptr;

        size_t ext_len = len < MIN_DATA_LENGTH ? MIN_DATA_LENGTH : len;
        size_t alloc_len = sizeof(len_head_t) + ext_len;
        size_t padding_len = 0;

        size_t mem_block_count = _mem_blocks.size();
        len_head_t* off_addr = (len_head_t*)addr(off);
        offset_t cur_off = 0;
        if (mem_block_count > 0) {
            cur_off = (mem_block_count - 1) * BLOCK_SIZE + _mem_blocks.back().off;
        }
        if (off < cur_off) { // from free lists
            if (_doing_defrag) {
                // it's just for safety, and the procedure should never reach here
                return alloc(try_alloc(len), len);
            }
            // free node no padding
            size_t free_node_len = sizeof(len_head_t) +
                    (((FreeNode*)off_addr)->len & DATA_LENGTH_MASK);
            offset_t next_off = ((FreeNode*)off_addr)->next_off;
            if (INVALID_OFFSET == next_off) {
                assert(off == *_head_of_off && "free list iterator abnormal");
                _sorted_free_sizes.erase(free_node_len);
                _free_lists.erase(free_node_len);
            } else {
                // update head with next_off after the head node in free list is consumed
                *_head_of_off = next_off;
            }

            --_free_node_cnt;

            if (free_node_len > alloc_len) {
                if (free_node_len - alloc_len < MIN_ALLOC_LENGTH) {
                    padding_len = free_node_len - alloc_len;
                } else {
                    dealloc_internal(off + alloc_len, free_node_len - alloc_len);
                }
            }
        } else {
            size_t idx = off / BLOCK_SIZE;
            if (mem_block_count == idx) { // new block
                if (mem_block_count > 0) {
                    // put left space in current block into free list
                    auto& last_block = _mem_blocks.back();
                    if (last_block.off < BLOCK_SIZE) {
                        put_into_free_list(cur_off, BLOCK_SIZE - last_block.off);
                        last_block.off = BLOCK_SIZE;
                    }
                }

                // create this new block
                off_addr = (len_head_t*)MmapUtility::malloc(BLOCK_SIZE);
                if (!off_addr) {
                    zmap_perror("malloc error");
                    return nullptr;
                }
                _mem_blocks.push_back({(uint8_t*)off_addr, 0});
            }

            size_t block_left_space = BLOCK_SIZE - (off + alloc_len) % BLOCK_SIZE;
            if (block_left_space < MIN_ALLOC_LENGTH) {
                padding_len = block_left_space;
                alloc_len += block_left_space;
            }

            _mem_blocks[idx].off += alloc_len;
        }

        *off_addr = len;
        *off_addr |= (padding_len << DATA_LENGTH_BIT_WIDTH);

        return off_addr + 1;
    }

    inline static size_t alloc_len(len_head_t len) {
        if (0 == len) return 0;
        size_t data_len = len & DATA_LENGTH_MASK;
        // 4-byte length header + extent data + rear padding
        return sizeof(len_head_t) + (data_len < MIN_DATA_LENGTH ? MIN_DATA_LENGTH : data_len)
                + ((len & PADDING_LENGTH_MASK) >> DATA_LENGTH_BIT_WIDTH);
    }

    void dealloc(offset_t off) {
        len_head_t* off_addr = (len_head_t*)addr(off);
        if (off_addr) {
            dealloc_internal(off, alloc_len(*off_addr));
        }
    }

    void* realloc(offset_t old_off, size_t new_data_len, offset_t* new_off = nullptr) {
        if (0 == new_data_len) {
            dealloc(old_off);
            return nullptr;
        }

        len_head_t* off_addr = (len_head_t*)addr(old_off);
        size_t old_alloc_len = off_addr ? alloc_len(*off_addr) : 0;
        size_t new_alloc_len = sizeof(len_head_t) + (new_data_len > MIN_DATA_LENGTH ?
                new_data_len : MIN_DATA_LENGTH);

        if (new_alloc_len > old_alloc_len) {
            offset_t off = INVALID_OFFSET;
            if (new_off && INVALID_OFFSET != *new_off) {
                off = *new_off;
            } else {
                off = try_alloc(new_data_len);
            }
            if (INVALID_OFFSET == off) return nullptr;
            auto addr = alloc(off, new_data_len);
            if (!addr) return nullptr;
            dealloc(old_off);
            if (new_off) *new_off = off;
            return addr;
        }

        *off_addr = new_data_len;

        if (old_alloc_len < new_alloc_len + MIN_ALLOC_LENGTH) {
            // add as padding
            *off_addr |= ((old_alloc_len - new_alloc_len) << DATA_LENGTH_BIT_WIDTH);
        } else {
            dealloc_internal(old_off + new_alloc_len, old_alloc_len - new_alloc_len);
        }

        if (new_off) *new_off = old_off;
        return off_addr + 1;
    }

    // get real address and length from specified offset
    inline std::pair<void*, size_t> convert(offset_t off) const {
        len_head_t* off_addr = (len_head_t*)addr(off);
        if (!off_addr) return {nullptr, 0};
        return {off_addr + 1, *off_addr & DATA_LENGTH_MASK};
    }

    void defrag() {
        if (!_doing_defrag.test_and_set()) return;

        _last_free_node_cnt = _free_node_cnt; // stop being triggered repeatedly
        // reserve a snapshot
        auto mem_blocks = std::make_shared<decltype(_mem_blocks)>(_mem_blocks);
        auto mb = *mem_blocks;
        std::thread thread([mem_blocks, this]() {
#ifndef NDEBUG
            struct timespec tp_start;
            clock_gettime(CLOCK_REALTIME, &tp_start);
            size_t valid_data = 0;
            size_t valid_data_count = 0;
            size_t padding_data = 0;
            size_t ext_data = 0;
            size_t free_data = 0;
            size_t free_node_count = 0;
            size_t max_single_free_data = 0;
            size_t max_single_free_data_now = 0;
            size_t merged_free_node_cnt = 0;
#endif
            _free_node_cnt = 0;
            _free_lists.clear();
            _sorted_free_sizes.clear();
            for (size_t i = 0, n = mem_blocks->size(); i < n; ++i) {
                uint8_t* p = mem_blocks->at(i).block;
                uint8_t* p_end = p + mem_blocks->at(i).off;
                int free_off = -1;
                size_t j = 0;
                while (p < p_end && !_stop_defrag) {
                    len_head_t len = *(len_head_t*)p;
                    size_t actual_len = alloc_len(len);
                    if (0 == actual_len) {
                        assert(actual_len > 0 && "abnormal length in allocator");
                        free_off = -1;
                        break;
                    }
                    if (len & DATA_IN_FREE_LIST_FLAG) {
                        assert(0 == ((len & PADDING_LENGTH_MASK) >> DATA_LENGTH_BIT_WIDTH) &&
                                "free node shouldn't have padding");
#ifndef NDEBUG
                        if (!(i + 1 == n && j + actual_len == BLOCK_SIZE)) {
                            max_single_free_data = std::max(max_single_free_data, actual_len);
                        }
#endif
                        if (-1 == free_off) {
                            free_off = j;
                        } else {
#ifndef NDEBUG
                            ++merged_free_node_cnt;
#endif
                        }
                    } else {
#ifndef NDEBUG
                        ++valid_data_count;
                        valid_data += actual_len;
                        padding_data += ((len & PADDING_LENGTH_MASK) >> DATA_LENGTH_BIT_WIDTH);
                        ext_data += (MIN_DATA_LENGTH > (len & DATA_LENGTH_MASK) ?
                                MIN_DATA_LENGTH - (len & DATA_LENGTH_MASK) : 0);
#endif
                        if (free_off >= 0) {
#ifndef NDEBUG
                            ++free_node_count;
                            free_data += j - free_off;
                            max_single_free_data_now = std::max(max_single_free_data_now,
                                    j - free_off);
#endif
                            put_into_free_list(i * BLOCK_SIZE + free_off, j - free_off, true);
                            free_off = -1;
                        }
                    }
                    j += actual_len;
                    p += actual_len;

                    if (p > p_end) {
                        (*mem_blocks)[i].off = j;
                    }
                }

                if (free_off >= 0) {
#ifndef NDEBUG
                    ++free_node_count;
                    free_data += j - free_off;
                    max_single_free_data_now = std::max(max_single_free_data_now, j - free_off);
#endif
                    put_into_free_list(i * BLOCK_SIZE + free_off,
                            mem_blocks->at(i).off - free_off, true);
                }
            }

#ifndef NDEBUG
            struct timespec tp_end;
            clock_gettime(CLOCK_REALTIME, &tp_end);
            printf("\n"
                   "defrag results:\n"
                   "\tblock: %lu\n"
                   "\tscaned data: %lu Bytes\n"
                   "\tvalid data: %lu Bytes\n"
                   "\tvalid data count: %lu\n"
                   "\tpadding data: %lu Bytes\n"
                   "\textended small data: %lu Bytes\n"
                   "\tfree_data: %lu Bytes\n"
                   "\tlast_block_offset: %lu\n"
                   "\tmerged free node: %lu\n"
                   "\tfree node now: %lu\n"
                   "\tmax single free data before: %lu Bytes\n"
                   "\tavg single free data before: %lu Bytes\n"
                   "\tmax single free data now: %lu Bytes\n"
                   "\tavg single free data now: %lu Bytes\n"
                   "\tfree list bucket count: %lu\n"
                   "\tfree nodes waiting to be processed: %lu\n"
                   "\n"
                   "\tspent %lu us\n"
                   "\n",
                   mem_blocks->size(), mem_blocks->size() * BLOCK_SIZE, valid_data,
                   valid_data_count, padding_data, ext_data, free_data,
                   mem_blocks->empty() ? BLOCK_SIZE : mem_blocks->back().off,
                   merged_free_node_cnt, free_node_count, max_single_free_data,
                   merged_free_node_cnt + free_node_count ? free_data /
                           (merged_free_node_cnt + free_node_count) : 0,
                   max_single_free_data_now, free_node_count ? free_data / free_node_count :
                           0, _free_lists.size(), _interim_free_nodes.size(),
                   tp_end.tv_sec * 1000 * 1000L + tp_end.tv_nsec / 1000 -
                   tp_start.tv_sec * 1000 * 1000L - tp_start.tv_nsec / 10000);
            assert((_stop_defrag || (mem_blocks->size() * BLOCK_SIZE == valid_data + free_data +
                    (BLOCK_SIZE - (mem_blocks->empty() ? BLOCK_SIZE :
                            mem_blocks->back().off)))) && "abnormal binary");
#endif

            _last_free_node_cnt = _free_node_cnt;
            std::lock_guard<std::mutex> lock(_interim_free_nodes_mtx);
            _doing_defrag.clear(); // it should be within lock
            for (const auto& node : _interim_free_nodes) {
                put_into_free_list(node.first, node.second);
            }
            _interim_free_nodes.clear();
        });
        thread.detach();
    }

    bool init() {
        if (!_free_lists.init(FREE_LISTS_CAPACITY)) {
            return false;
        }
        _mem_blocks.reserve(10000);
        return true;
    }

    bool load(const std::string& path) {
        auto&& baton = FileIOUtility::mload(path);

        _mem_blocks.reserve(10000);
        for (size_t off = 0, sz = baton.size(); off < sz && !baton.eof(); off += BLOCK_SIZE) {
            uint8_t* addr = (uint8_t*)MmapUtility::malloc(BLOCK_SIZE);
            if (!addr) {
                zmap_perror("malloc error");
                return false;
            }

            if (!baton.read(addr, BLOCK_SIZE)) return false;
            _mem_blocks.push_back({addr, BLOCK_SIZE});
        }

        if (!_free_lists.init(FREE_LISTS_CAPACITY)) return false;

        defrag();
        return true;
    }

    bool dump(const std::string& path, DumpWriter* writer = nullptr) {
        if (_mem_blocks.empty()) return true;

        // mark left space in current block as free, so as to calc cur_block.off when load
        auto& cur_block = _mem_blocks.back();
        *(len_head_t*)(cur_block.block + cur_block.off) =
                (BLOCK_SIZE - cur_block.off - sizeof(len_head_t)) | DATA_IN_FREE_LIST_FLAG;

        if (writer) {
            if (!writer->init_data(path)) return false;
            for (const auto& mem_block : _mem_blocks) {
                if (BLOCK_SIZE != writer->write_data(mem_block.block, BLOCK_SIZE)) {
                    return false;
                }
            }
        } else {
            auto&& baton = FileIOUtility::mdump(path);
            if (!baton) return false;

            for (const auto& mem_block : _mem_blocks) {
                if (!baton.write(mem_block.block, BLOCK_SIZE)) return false;
            }
        }
        return true;
    }

    void debug(std::ostream* out = nullptr) const {
        std::ostringstream oss;
        std::ostream* pout = out ? out : &oss;
        *pout << "\n[alloc] debug begin {\n";
        *pout << string_printf("\tmem blocks count: %lu, block size: %lu\n",
                _mem_blocks.size(), BLOCK_SIZE);
        for (size_t i = 0; i < _mem_blocks.size(); ++i) {
            *pout << string_printf("\tmem[%lu].off = %lu\n", i, _mem_blocks[i].off);
        }
        *pout << "\n";
        size_t total_free_size = 0;
        std::lock_guard<std::mutex> lock(_free_lists_mtx);
        *pout << string_printf("\tfree node count: %lu\n", _free_node_cnt.load());
        for (auto free_len : _sorted_free_sizes) {
            const void* free_len_val = nullptr;
            offset_t off = INVALID_OFFSET;
            if (_free_lists.find_ex(free_len, free_len_val)) {
                off = *(offset_t*)free_len_val;
            } else {
                continue;
            }

            size_t n = 1;
            while (true) {
                FreeNode* free_node = (FreeNode*)addr(off);
                if (INVALID_OFFSET == free_node->next_off) {
                    break;
                }
                ++n;
                off = free_node->next_off;
            }
            total_free_size += free_len * n;
            *pout << string_printf("\tfree_len: %u, count: %lu, free_size: %lu\n",
                    free_len, n, free_len * n);
        }
        *pout << string_printf("\ttotal_free_size: %lu\n", total_free_size);
        *pout << "} alloc debug end\n\n";
        if (!out) printf("%s", oss.str().data());
    }

private:
    inline void* addr(offset_t off) const {
        if (INVALID_OFFSET == off) return nullptr;
        auto idx = off / BLOCK_SIZE;
        if (idx >= _mem_blocks.size()) return nullptr;
        return _mem_blocks[idx].block + off % BLOCK_SIZE;
    }

    inline void dealloc_internal(offset_t off, size_t free_len) {
        if (INVALID_OFFSET == off || 0 == free_len) return;
        auto idx = off / BLOCK_SIZE;
        // no need to put into free list if returned node is just allocated
        // from the current mem block
        if (_mem_blocks.size() == idx + 1 &&
                _mem_blocks[idx].off == (off % BLOCK_SIZE) + free_len) {
            _mem_blocks[idx].off -= free_len;
            return;
        }
        put_into_free_list(off, free_len);
    }

    void put_into_free_list(offset_t off, size_t free_len, bool put_forcedly = false) {
        if (INVALID_OFFSET == off || 0 == free_len) return;
        assert(free_len >= MIN_ALLOC_LENGTH && "free_len less than MIN_ALLOC_LENGTH");

        if (!put_forcedly && _doing_defrag) {
            std::unique_lock<std::mutex> lock(_interim_free_nodes_mtx, std::try_to_lock);
            if (lock && _doing_defrag) { // lock successfully, i.e. own the lock
                _interim_free_nodes.emplace_back(off, free_len);
                return;
            }
        }

        void* off_addr = addr(off);
        FreeNode* new_head = (FreeNode*)off_addr;
        new_head->len = (free_len - sizeof(len_head_t)) | DATA_IN_FREE_LIST_FLAG;

        ++_free_node_cnt;

        std::lock_guard<std::mutex> lock(_free_lists_mtx);
        new_head->next_off = INVALID_OFFSET;
        auto ret = _free_lists.emplace(free_len, &off, sizeof(offset_t));
        if (!ret.second) {
            auto last_head = (offset_t*)ret.first.raw_slot()->val;
            new_head->next_off = *last_head;
            *last_head = off; // update head off with the new one
        }
    }

private:
    // (free_item_size, free_list)
    ZmapImpl<ProbeGroup128, sizeof(offset_t), DefaultHasher> _free_lists;
    mutable std::mutex _free_lists_mtx;
    std::vector<std::pair<offset_t, size_t>> _interim_free_nodes; // store free nodes during defrag
    std::mutex _interim_free_nodes_mtx;

    offset_t* _head_of_off = nullptr;
    std::set<uint32_t> _sorted_free_sizes;

    std::atomic<size_t> _free_node_cnt = {0};
    size_t _last_free_node_cnt = 0;
    atomic_flag _doing_defrag;
    volatile bool _stop_defrag = false;
    std::vector<MemBlock> _mem_blocks;
    constexpr static size_t FREE_LISTS_CAPACITY = 1024 * 1024;
    constexpr static size_t DATA_IN_FREE_LIST_FLAG = 1U << 31;
    constexpr static size_t DATA_LENGTH_BIT_WIDTH = 27;
    constexpr static size_t DATA_LENGTH_MASK = (1 << DATA_LENGTH_BIT_WIDTH) - 1;
    constexpr static size_t PADDING_LENGTH_MASK =
            (len_head_t)~(DATA_IN_FREE_LIST_FLAG | DATA_LENGTH_MASK);
    constexpr static size_t BLOCK_SIZE = 1UL << DATA_LENGTH_BIT_WIDTH;
    constexpr static size_t MIN_DATA_LENGTH = 8;
    constexpr static size_t MAX_DATA_LENGTH = BLOCK_SIZE - sizeof(len_head_t);
    constexpr static size_t MIN_ALLOC_LENGTH = sizeof(len_head_t) + MIN_DATA_LENGTH;
};

// support value with variable length
template <typename ProbeGroupType, typename Hasher = DefaultHasher, typename Alloc = DefaultAlloc>
class ElasticZmapImpl {
    using ZmapType = ZmapImpl<ProbeGroupType, sizeof(offset_t), Hasher>;
    ZmapType _zmap;
    Alloc _alloc;
    using Slot = std::pair<uint64_t, std::string>;
    template <typename, size_t> friend struct ParallelZmapImpl;
    template <typename, typename> friend class StringZmapImpl;

    // considering atomicity of _zmap.emplace, we have to prepare value in advance,
    // and dealloc it if key isn't inserted successfully
    struct ValueWrapper {
        ValueWrapper(Alloc* alloc, const void* val, size_t val_len): _alloc(alloc) {
            if (!val || 0 == val_len) {
                val = nullptr;
                val_len = 0;
            }
            if (val) {
                _off = _alloc->try_alloc(val_len);
                if (INVALID_OFFSET != _off) {
                    auto addr = _alloc->alloc(_off, val_len);
                    if (addr) {
                        memcpy(addr, val, val_len);
                        return;
                    }
                }
                _alloc = nullptr;
                _off = INVALID_OFFSET;
            }
        }
        ~ValueWrapper() {
            if (_alloc && INVALID_OFFSET != _off) _alloc->dealloc(_off);
        }

        inline explicit operator bool() const { return _alloc; }
        inline const offset_t* get_offset() const { return &_off; }
        inline void dismiss() { _off = INVALID_OFFSET; }

        Alloc* _alloc = nullptr;
        offset_t _off = INVALID_OFFSET;
    };

public:
    struct const_iterator {
        const_iterator() = default;
        const_iterator& operator=(const const_iterator& other) = default;
        inline const Slot& operator*() const {
            // use the lazy mode to init local Slot for performance,
            // but the following paradox might occur:
            // auto iter = m.find(key);
            // if (m.end() != iter) {
            //     iter->second; // value is empty once key might have already been deleted
            // }
            if (_slot_inited_flag.test_and_set()) {
                _slot.first = -1;
                _slot.second = "";
                auto lock = _iter.template lock<false>();
                auto raw_slot = _iter.raw_slot();
                if (raw_slot) {
                    _slot.first = raw_slot->first;
                    auto val = _ezmap->convert(raw_slot->second);
                    if (val.first && val.second > 0) {
                        _slot.second.assign((const char*)val.first, val.second);
                    }
                }
            }
            return _slot;
        }
        inline const Slot* operator->() const { return &operator*(); }

        inline const_iterator& operator++() {
            ++_iter;
            _slot_inited_flag.clear();
            return *this;
        }

        inline friend bool operator==(const const_iterator& a, const const_iterator& b) {
            return a._iter == b._iter;
        }
        inline friend bool operator!=(const const_iterator& a, const const_iterator& b) {
            return !(a == b);
        }

    private:
        friend class ElasticZmapImpl;
        const_iterator(typename ZmapType::const_iterator iter, const ElasticZmapImpl* ezmap):
            _iter(iter), _ezmap(ezmap) {}
        auto raw() const { return _iter; }
        typename ZmapType::const_iterator _iter;
        const ElasticZmapImpl* _ezmap = nullptr;
        mutable Slot _slot;
        mutable atomic_flag _slot_inited_flag;
    };

    typedef uint64_t KeyType;
    typedef typename std::remove_cv<typename std::remove_reference<KeyType>::type>::type RawKeyType;
    typedef Hasher HasherType;
    ElasticZmapImpl() {};

    ElasticZmapImpl(uint64_t max_size) {
        bool init_result = init(max_size);
        assert(init_result && "init failed");
        (void)init_result;
    }

    ElasticZmapImpl(const std::string& path, bool use_mmap = false) {
        bool load_result = load(path, use_mmap);
        assert(load_result && "load failed");
        (void)load_result;
    }

    bool init(uint64_t max_size, size_t sub_idx = -1) {
        return _zmap.init(max_size) && _alloc.init();
    }

    bool load(const std::string& path, bool use_mmap = false) {
        return _zmap.load(get_zmap_path(path, {".zmap_idx"}), use_mmap)
                && _alloc.load(get_zmap_path(path, {".zmap_val"}));
    }

    // the filename saved finally will add a suffix with ".zmap_idx"(for index part)
    // and ".zmap_val"(for value part) if necessary
    bool dump(const std::string& path, size_t sub_idx = -1, DumpWriter* writer = nullptr) {
        return _zmap.dump(get_zmap_path(path, {".zmap_idx"}), sub_idx, writer)
                && _alloc.dump(get_zmap_path(path, {".zmap_val"}), writer);
    }

    bool unlink(const std::string& path) {
        bool ret = (0 == ::unlink(get_zmap_path(path, {".zmap_idx"}).data()));
        ret &= (0 == ::unlink(get_zmap_path(path, {".zmap_val"}).data()));
        return ret;
    }
    inline static size_t subcnt() { return 1UL; }
    inline static size_t subidx(size_t hash_val) { return 0UL; }

    inline const_iterator begin() const { return {_zmap.begin(), this}; }
    inline const_iterator end() const { return {_zmap.end(), this}; }

    inline std::pair<const_iterator, bool> emplace(uint64_t key, const void* val,
            size_t val_len, size_t hash_val = -1) {
        ValueWrapper vw(&_alloc, val, val_len);
        if (vw) {
            auto ret = _zmap.emplace(key, vw.get_offset(), 0, hash_val);
            if (ret.second) vw.dismiss();
            return {{ret.first, this}, ret.second};
        }
        return {end(), false};
    }
    inline std::pair<const_iterator, bool> emplace_s(uint64_t key, const std::string& val,
            size_t hash_val = -1) {
        return emplace(key, val.data(), val.size(), hash_val);
    }
    // the implementation of const_iterator will construct a string object via copy value,
    // and emplace_ex could avoid the copy of it.
    // return -1 if emplace failed, e.g. map is full
    // return 0 if key already exists
    // return 1 if insert ok
    inline int emplace_ex(uint64_t key, const void* val, size_t val_len, size_t hash_val = -1) {
        ValueWrapper vw(&_alloc, val, val_len);
        if (vw) {
            auto ret = _zmap.emplace_ex(key, vw.get_offset(), 0, hash_val);
            if (1 == ret) vw.dismiss();
            return ret;
        }
        return -1;
    }

    inline std::pair<const_iterator, bool> insert(const std::pair<uint64_t, std::string>& kv,
            size_t hash_val = -1) {
        return emplace(kv.first, kv.second.data(), kv.second.size(), hash_val);
    }

#define EZMAP_UPDATE(abnormal_ret)                                                       \
        auto ret = _zmap.emplace(key, nullptr, 0, hash_val);                             \
        if (_zmap.end() == ret.first && !ret.second) return abnormal_ret;                \
                                                                                         \
        do {                                                                             \
            offset_t old_off = INVALID_OFFSET;                                           \
            if (!ret.second) {                                                           \
                auto lock = ret.first.template lock<true>();                             \
                auto raw_slot = ret.first.raw_slot();                                    \
                                                                                         \
                if (val && val_len > 0) {                                                \
                    auto old_val = convert(raw_slot->second);                            \
                    if (old_val.first && old_val.second == val_len) {                    \
                        memcpy(old_val.first, val, val_len);                             \
                        break;                                                           \
                    }                                                                    \
                }                                                                        \
                                                                                         \
                const void* poff = raw_slot->second;                                     \
                old_off = *(const offset_t*)(poff);                                      \
            }                                                                            \
                                                                                         \
            ValueWrapper vw(&_alloc, val, val_len);                                      \
            if (!vw) return abnormal_ret;                                                \
            {                                                                            \
                auto lock = ret.first.template lock<true>();                             \
                memcpy(ret.first.raw_slot()->second, vw.get_offset(), sizeof(offset_t)); \
            }                                                                            \
            vw.dismiss();                                                                \
            _alloc.dealloc(old_off);                                                     \
        } while (0)

    // ${return_val}.second is true if key doesn't exist before
    inline std::pair<const_iterator, bool> update(uint64_t key, const void* val, size_t val_len,
            size_t hash_val = -1) {
        EZMAP_UPDATE((std::pair<const_iterator, bool>{end(), false}));
        return {{ret.first, this}, ret.second};
    }
    inline std::pair<const_iterator, bool> update_s(uint64_t key, const std::string& val,
            size_t hash_val = -1) {
        return update(key, val.data(), val.size(), hash_val);
    }
    inline int update_ex(uint64_t key, const void* val, size_t val_len, size_t hash_val = -1) {
        EZMAP_UPDATE(-1);
        return ret.second;
    }

    template <bool NO_DUP_KEY = false>
    inline int build(uint64_t key, const void* val, size_t val_len = 0, size_t hash_val = -1) {
        ValueWrapper vw(&_alloc, val, val_len);
        if (vw) {
            auto ret = _zmap.template build<NO_DUP_KEY>(key, vw.get_offset(), 0, hash_val);
            if (ZMAP_PREDICT_TRUE(1 == ret)) vw.dismiss();
            return ret;
        }
        return -1;
    }

    inline size_t erase(uint64_t key, size_t hash_val = -1) {
        return erase_internal(_zmap.find(key, hash_val));
    }
    inline size_t erase(const const_iterator& it) {
        return erase_internal(it.raw());
    }

    inline const_iterator find(uint64_t key, size_t hash_val = -1) const {
        return {_zmap.find(key, hash_val), this};
    }
    // the implementation of const_iterator will construct a string object via copy value,
    // so find_ex could provide the pointer of value directly, but you should take the risk
    // that the content of pointer might be modified and you'd better just read it without
    // any modification.
    inline bool find_ex(uint64_t key, const void*& val, size_t* val_len,
            size_t hash_val = -1) const {
        auto iter = _zmap.find(key, hash_val);
        if (_zmap.end() != iter) {
            auto ret = convert(iter->second);
            val = ret.first;
            if (val_len) {
                *val_len = ret.second;
            }
            return true;
        }
        return false;
    }

    inline size_t count(uint64_t key, size_t hash_val = -1) const {
        return _zmap.count(key, hash_val);
    }

    inline const std::string at(uint64_t key, size_t hash_val = -1) const {
        auto iter = find(key, hash_val);
        if (end() == iter) abort();
        return iter->second;
    }

    // insert key with empty value if key doesn't exist in map,
    // but we couldn't return an reference of value
    inline const std::string operator[](uint64_t key) {
        auto ret = emplace(key, nullptr, 0);
        return ret.first->second;
    }

    // just facilitate for test
    inline uint64_t rand_key() const { return _zmap.rand_key(); }
    inline size_t size() const { return _zmap.size(); }
    inline size_t capacity() const { return _zmap.capacity(); }
    inline bool empty() const { return _zmap.empty(); }
    inline void clear() { _zmap.clear(); _alloc.clear(); }
    inline void reset() { _zmap.reset(); _alloc.reset(); }
    void debug(std::ostream* out = nullptr, size_t sub_idx = -1) {
        std::ostringstream oss;
        std::ostream* pout = out ? out : &oss;
        *pout << "[elastic zmap]\n";
        _zmap.debug(pout);
        _alloc.debug(pout);
        if (!out) printf("%s", oss.str().data());
    }

private:
    // convert offset to addr
    inline std::pair<void*, size_t> convert(const void* val_off_ptr) const {
        return _alloc.convert(*(offset_t*)val_off_ptr);
    }

    inline size_t erase_internal(const typename ZmapType::const_iterator& raw_it) {
        const void* poff = raw_it->second;
        auto old_off = *(const offset_t*)(poff);
        auto ret = _zmap.erase(raw_it);
        if (ret) _alloc.dealloc(old_off);
        return ret;
    }

    inline auto& header() { return _zmap.header(); }
    inline const auto& header() const { return _zmap.header(); }
};

// support key & value with string
template <typename ProbeGroupType, typename Alloc = DefaultAlloc>
class StringZmapImpl {
    struct IdentityHasher {
        inline uint64_t operator()(uint64_t key) { return key; }
    };
    struct KeyHasher {
        inline uint64_t operator()(const std::string& key) {
            return MurmurHash64A(key.data(), key.size());
        }
    };
    using ZmapType = ElasticZmapImpl<ProbeGroupType, IdentityHasher, Alloc>;
    using Slot = std::pair<std::string, std::string>;
    struct SubValueNode {
        uint32_t key_len;
        uint32_t val_len;
        char key[0];
    };
    template <typename, size_t> friend struct ParallelZmapImpl;

    ZmapType _zmap;
    std::atomic<uint64_t> _conflict_keys_cnt = {0};

    struct const_iterator {
        const_iterator() = default;
        const_iterator& operator=(const const_iterator& other) = default;
        inline const Slot& operator*() const {
            // use the lazy mode to init local Slot for performance,
            if (_slot_inited_flag.test_and_set()) {
                _raw_val = _iter->second;
                if (_raw_val.empty()) { // end() or key has been erased
                    _sub_val_idx = -1;
                    return _slot;
                }
                for (size_t n = 0, sz = _raw_val.size(), i = 0; n < sz; ++i) {
                    auto* sub_val = (const SubValueNode*)(_raw_val.data() + n);
                    n += sizeof(SubValueNode) + sub_val->key_len + sub_val->val_len;
                    if (n > sz) {
                        _sub_val_idx = -1;
                        _slot = {};
                        zmap_perror("sub value overflow");
                        return _slot;
                    }

                    if (i == (size_t)_sub_val_idx) {
                        _slot.first.assign(sub_val->key, sub_val->key_len);
                        _slot.second.assign(sub_val->key + sub_val->key_len, sub_val->val_len);
                        _next_val_ptr = _raw_val.data() + n;
                        if (n == sz) {
                            _sub_val_idx = -1;
                        }
                        break;
                    }
                }
            }

            return _slot;
        }
        inline const Slot* operator->() const { return &operator*(); }

        inline const_iterator& operator++() {
            if (!_next_val_ptr) operator*();
            if (ZMAP_PREDICT_TRUE(_sub_val_idx < 0)) {
                ++_iter;
                _sub_val_idx = 0;
                _next_val_ptr = nullptr;
                _slot_inited_flag.clear();
            } else {
                if (_next_val_ptr) {
                    ++_sub_val_idx;
                    auto* sub_val = (const SubValueNode*)_next_val_ptr;
                    _next_val_ptr += sizeof(SubValueNode) + sub_val->key_len + sub_val->val_len;
                    if (_next_val_ptr > _raw_val.data() + _raw_val.size()) {
                        _sub_val_idx = -1;
                        _slot = {};
                        zmap_perror("sub value overflow");
                        return *this;
                    }
                    _slot.first.assign(sub_val->key, sub_val->key_len);
                    _slot.second.assign(sub_val->key + sub_val->key_len, sub_val->val_len);

                    if (_next_val_ptr == _raw_val.data() + _raw_val.size()) {
                        _sub_val_idx = -1;
                    }
                }
            }
            return *this;
        }

        inline friend bool operator==(const const_iterator& a, const const_iterator& b) {
            return a._iter == b._iter && a._sub_val_idx == b._sub_val_idx;
        }
        inline friend bool operator!=(const const_iterator& a, const const_iterator& b) {
            return !(a == b);
        }

    private:
        friend class StringZmapImpl;
        const_iterator(typename ZmapType::const_iterator iter, int sub_val_idx = 0):
            _iter(iter), _sub_val_idx(sub_val_idx) {}
        typename ZmapType::const_iterator _iter;
        mutable Slot _slot;
        mutable std::string _raw_val;
        mutable const char* _next_val_ptr = nullptr;
        mutable int _sub_val_idx = 0;
        mutable atomic_flag _slot_inited_flag;
    };

    enum ModifyType { MT_INSERT, MT_UPDATE, MT_ERASE };

public:
    typedef const std::string& KeyType;
    typedef typename std::remove_cv<typename std::remove_reference<KeyType>::type>::type RawKeyType;
    typedef KeyHasher HasherType;
    StringZmapImpl() {};

    StringZmapImpl(uint64_t max_size) {
        bool init_result = init(max_size);
        assert(init_result && "init failed");
        (void)init_result;
    }

    StringZmapImpl(const std::string& path, bool use_mmap = false) {
        bool load_result = load(path, use_mmap);
        assert(load_result && "load failed");
        (void)load_result;
    }

    bool init(uint64_t max_size, size_t sub_idx = -1) {
        return _zmap.init(max_size, sub_idx);
    }

    bool load(const std::string& path, bool use_mmap = false) {
        bool ret = _zmap.load(path, use_mmap);
        auto& header = _zmap.header();
        _conflict_keys_cnt = (((uint16_t)header.reserve2) << 8) + header.reserve3;
        return ret;
    }

    // the filename saved finally will add a suffix with ".zmap_idx"(for index part)
    // and ".zmap_val"(for value part) if necessary
    bool dump(const std::string& path, size_t sub_idx = -1, DumpWriter* writer = nullptr) {
        if (_conflict_keys_cnt < 65535) {
            auto& header = _zmap.header();
            header.reserve2 = _conflict_keys_cnt >> 8;
            header.reserve3 = _conflict_keys_cnt & 0xFF;
        }
        return _zmap.dump(path, sub_idx, writer);
    }

    bool unlink(const std::string& path) {
        return _zmap.unlink(path);
    }
    inline static size_t subcnt() { return 1UL; }
    inline static size_t subidx(size_t hash_val) { return 0UL; }

    inline const_iterator begin() const { return {_zmap.begin()}; }
    inline const_iterator end() const { return {_zmap.end()}; }

    inline std::pair<const_iterator, bool> emplace(const std::string& key, const void* val,
            size_t val_len, size_t hash_val = -1) {
        return modify<MT_INSERT>(key, val, val_len, hash_val);
    }
    inline std::pair<const_iterator, bool> emplace_s(const std::string& key,
            const std::string& val, size_t hash_val = -1) {
        return emplace(key, val.data(), val.size(), hash_val);
    }
    // return -1 if emplace failed, e.g. map is full
    // return 0 if key already exists
    // return 1 if insert ok
    inline int emplace_ex(const std::string& key, const void* val, size_t val_len,
            size_t hash_val = -1) {
        auto ret = emplace(key, val, val_len, hash_val);
        if (ret.first == end() && ret.second == false) return -1;
        return ret.second;
    }

    inline std::pair<const_iterator, bool> insert(const std::pair<std::string,
            std::string>& kv, size_t hash_val = -1) {
        return emplace(kv.first, kv.second.data(), kv.second.size(), hash_val);
    }

    // ${return_val}.second is true if key doesn't exist before
    inline std::pair<const_iterator, bool> update(const std::string& key, const void* val,
            size_t val_len, size_t hash_val = -1) {
        return modify<MT_UPDATE>(key, val, val_len, hash_val);
    }
    inline std::pair<const_iterator, bool> update_s(const std::string& key,
            const std::string& val, size_t hash_val = -1) {
        return update(key, val.data(), val.size(), hash_val);
    }
    inline int update_ex(const std::string& key, const void* val, size_t val_len,
            size_t hash_val = -1) {
        auto ret = update(key, val, val_len, hash_val);
        if (ret.first == end() && ret.second == false) return -1;
        return ret.second;
    }

    template <bool NO_DUP_KEY = false>
    inline int build(const std::string& key, const void* val, size_t val_len = 0,
            size_t hash_val = -1) {
        return update_ex(key, val, val_len, hash_val);
    }

    inline size_t erase(const std::string& key, size_t hash_val = -1) {
        return modify<MT_ERASE>(key, nullptr, 0, hash_val).second;
    }
    inline size_t erase(const const_iterator& it) {
        return erase(it->first);
    }

    inline const_iterator find(const std::string& key, size_t hash_val = -1) const {
        if ((size_t)-1 == hash_val) hash_val = MurmurHash64A(key.data(), key.size());
        auto iter = _zmap.find(hash_val, hash_val);
        if (_zmap.end() == iter) return iter;
        const_iterator ret_iter(iter);
        while (ZMAP_PREDICT_FALSE(key != ret_iter->first)) {
            if (!ret_iter._next_val_ptr || ret_iter._sub_val_idx < 0) return end();
            ++ret_iter;
        }
        return ret_iter;
    }
    // the implementation of const_iterator will construct a string object via copy value,
    // so find_ex could provide the pointer of value directly, but you should take the risk
    // that the content of pointer might be modified and you'd better just read it without
    // any modification.
    inline bool find_ex(const std::string& key, const void*& val, size_t* val_len,
            size_t hash_val = -1) const {
        if ((size_t)-1 == hash_val) hash_val = MurmurHash64A(key.data(), key.size());

        const void* raw_val = nullptr;
        size_t raw_val_len = 0;
        if (_zmap.find_ex(hash_val, raw_val, &raw_val_len, hash_val)) {
            for (size_t n = 0; n < raw_val_len;) {
                auto* sub_val = (const SubValueNode*)((const char*)raw_val + n);
                // find the key
                if (ZMAP_PREDICT_TRUE(0 == memcmp(key.data(), sub_val->key, sub_val->key_len))) {
                    val = sub_val->key + sub_val->key_len;
                    if (val_len) *val_len = sub_val->val_len;
                    return true;
                }
                n += sizeof(SubValueNode) + sub_val->key_len + sub_val->val_len;
                assert(n <= raw_val_len && "sub value overflow");
            }
        }
        return false;
    }

    inline size_t count(const std::string& key, size_t hash_val = -1) const {
        const void* val = nullptr;
        return find_ex(key, val, nullptr, hash_val);
    }

    inline const std::string at(const std::string& key, size_t hash_val = -1) const {
        const void* val = nullptr;
        size_t val_len = 0;
        if (!find_ex(key, val, &val_len, hash_val)) {
            abort();
        }
        return std::string((const char*)val, val_len);
    }

    // insert key with empty value if key doesn't exist in map,
    // but we couldn't return an reference of value
    inline const std::string operator[](const std::string& key) {
        auto ret = emplace(key, nullptr, 0);
        return ret.first->second;
    }

    // just make it easy for test
    inline std::string rand_key() const {
        while (!_zmap.empty()) {
            auto iter = _zmap.find(_zmap.rand_key());
            if (_zmap.end() != iter && (uint64_t)-1 != iter->first) {
                return const_iterator(iter)->first;
            }
        }
        return "";
    }
    inline size_t size() const { return _zmap.size() + _conflict_keys_cnt; }
    inline size_t capacity() const { return _zmap.capacity() + _conflict_keys_cnt; }
    inline bool empty() const { return _zmap.empty(); }
    inline void clear() {
        _zmap.clear();
        _conflict_keys_cnt = {0};
    }
    inline void reset() {
        _zmap.reset();
        _conflict_keys_cnt = {0};
    }
    void debug(std::ostream* out = nullptr, size_t sub_idx = -1) {
        std::ostringstream oss;
        std::ostream* pout = out ? out : &oss;
        *pout << "[string zmap]\n";
        _zmap.debug(pout);
        if (!out) printf("%s", oss.str().data());
    }

private:
    template <ModifyType MTYPE = MT_INSERT>
    inline std::pair<const_iterator, bool> modify(const std::string& key, const void* val,
            size_t val_len, size_t hash_val = -1) {
        if ((size_t)-1 == hash_val) hash_val = MurmurHash64A(key.data(), key.size());
        std::string new_val;
        if (MT_ERASE != MTYPE) {
            SubValueNode new_sub_val = {(uint32_t)key.size(), (uint32_t)val_len, {}};
            new_val.append((const char*)&new_sub_val, sizeof(SubValueNode));
            new_val.append(key);
            new_val.append((const char*)val, val_len);
        }

        std::pair<typename ZmapType::const_iterator, bool> ret;
        if (MT_ERASE == MTYPE) {
            ret.first = _zmap.find(hash_val, hash_val);
            if (ret.first == _zmap.end()) {
                return {end(), false}; // erase failed because key doesn't exist
            }
            ret.second = (ret.first == _zmap.end()); // ret.second is false if key exists
        } else {
            ret = _zmap.emplace(hash_val, new_val.data(), new_val.size(), hash_val);
        }
        if (!ret.second) {
            if (ret.first == _zmap.end()) return {{ret.first}, false}; // zmap full

            const auto& existed_val = ret.first->second;
            int sub_val_idx = 0;
            for (size_t n = 0, sz = existed_val.size(); n < sz; ++sub_val_idx) {
                auto* sub_val = (const SubValueNode*)(existed_val.data() + n);
                // find the key
                if (ZMAP_PREDICT_TRUE(key == std::string(sub_val->key, sub_val->key_len))) {
                    if (MT_INSERT == MTYPE || (sub_val->val_len == val_len &&
                            0 == memcmp(sub_val->key + sub_val->key_len, val, val_len))) {
                        return {{ret.first, sub_val_idx}, false};
                    } else { // update value or erase key
                        if (MT_ERASE == MTYPE) {
                            new_val = existed_val.substr(0, n);
                        } else {
                            new_val = existed_val.substr(0, n) + new_val;
                        }
                        n += sizeof(SubValueNode) + sub_val->key_len + sub_val->val_len;
                        assert(n <= sz && "sub value overflow");
                        if (n < sz) {
                            new_val += existed_val.substr(n);
                        }
                        if (new_val.empty()) {
                            _zmap.erase(ret.first);
                            return {end(), true};
                        }
                        if (MT_ERASE == MTYPE) --_conflict_keys_cnt;
                        return {{_zmap.update_s(hash_val, new_val).first, sub_val_idx},
                            MT_ERASE == MTYPE};
                    }
                }
                n += sizeof(SubValueNode) + sub_val->key_len + sub_val->val_len;
                assert(n <= sz && "sub value overflow");
            }

            if (MT_ERASE == MTYPE) { // no matched key to be erased if program runs here
                return {end(), false};
            }

            new_val = existed_val + new_val;
            ++_conflict_keys_cnt;
            return {{_zmap.update_s(hash_val, new_val).first, sub_val_idx}, true};
        }
        return {{ret.first}, true};
    }

    inline auto& header() { return _zmap.header(); }
    inline const auto& header() const { return _zmap.header(); }
};

template <typename ZmapType, size_t N> // 2**N submaps
struct ParallelZmapImpl {
    typedef typename ZmapType::KeyType KeyType;
    typedef typename ZmapType::RawKeyType RawKeyType;
    typedef typename ZmapType::HasherType Hasher;
    ParallelZmapImpl() = default;
    ParallelZmapImpl(uint64_t max_size) {
        bool init_result = init(max_size);
        assert(init_result && "init failed");
        (void)init_result;
    }

    ParallelZmapImpl(const std::string& path, bool use_mmap = false) {
        bool load_result = load(path, use_mmap);
        assert(load_result && "load failed");
        (void)load_result;
    }

    bool init(uint64_t max_size, size_t sub_idx = -1) {
        static_assert(N <= 8, "N should be less than 8");
        memset(_spilled_zmap_count, 0, sizeof(_spilled_zmap_count));

        if (max_size < MIN_TABLE_SIZE) max_size = MIN_TABLE_SIZE;

        uint64_t sub_max_size = max_size >> N;
        uint64_t new_sub_max_size = calc_max_size(sub_max_size);

        // capacity keys will be hashed into 2**N zmaps, and considering data
        // skew, the factual capacity is expanded to capacity * sparse_factor.
        double sparse_factor = 1.25;
        if ((new_sub_max_size << N) < max_size * sparse_factor) {
            new_sub_max_size *= 2; // upgrade to next level
        }

        volatile bool ret = true;

        std::vector<std::thread> threads;
        for (size_t i = 0; i <= _mask; ++i) {
            if ((size_t)-1 != sub_idx && sub_idx != i) continue;
            threads.emplace_back([&ret, new_sub_max_size, i, this]() {
                if (ret && !_subs[i].init(new_sub_max_size)) {
                    ret = false;
                }
            });
        }
        for (auto& thread : threads) thread.join();

        return ret;
    }

    // it's not necessary that path ends with a suffix like ".zmap_*" or ".part-XXX".
    bool load(const std::string& path, bool use_mmap = false) {
        volatile bool ret = true;

        std::vector<std::thread> threads;
        for (size_t i = 0; i <= _mask; ++i) {
            threads.emplace_back([&ret, &path, use_mmap, i, this]() {
                if (ret && !_subs[i].load(get_zmap_path(path,
                        {string_printf(".part-%lu", i)}, true), use_mmap)) {
                    ret = false;
                }
                _spilled_zmap_count[i] = _subs[i].header().reserve1;
            });
        }
        for (auto& thread : threads) thread.join();

        return ret;
    }

    // the filename saved finally will add a suffix with ".part-XXX" for one part
    // if sub_idx is not -1, then just dump one sub map
    bool dump(const std::string& path, size_t sub_idx = -1, DumpWriter* writer = nullptr) {
        volatile bool ret = true;

        std::vector<std::thread> threads;
        for (size_t i = 0; i <= _mask; ++i) {
            if ((size_t)-1 != sub_idx && sub_idx != i) continue;
            threads.emplace_back([&ret, &path, i, sub_idx, writer, this]() {
                _subs[i].header().reserve1 = (uint8_t)_spilled_zmap_count[i];
                if (ret && !_subs[i].dump(get_zmap_path(path,
                        {string_printf(".part-%lu", i)}), sub_idx, writer)) {
                    ret = false;
                }
            });
        }
        for (auto& thread : threads) thread.join();

        return ret;
    }

    bool unlink(const std::string& path) {
        bool ret = true;
        for (size_t i = 0; i <= _mask; ++i) {
            ret &= _subs[i].unlink(get_zmap_path(path, {string_printf(".part-%lu", i)}).data());
        }
        return ret;
    }

    struct const_iterator {
        using Slot = typename ZmapType::Slot;
        const_iterator() = default;

        inline const Slot& operator*()  const { return *_it; }
        inline const Slot* operator->() const { return &operator*(); }

        inline const_iterator& operator++() {
            if (_inner) {
                ++_it;
                skip_empty();
            }
            return *this;
        }

        inline friend bool operator==(const const_iterator& a, const const_iterator& b) {
            return a._inner == b._inner && (!a._inner || a._it == b._it);
        }

        inline friend bool operator!=(const const_iterator& a, const const_iterator& b) {
            return !(a == b);
        }

    private:
        template <typename, size_t> friend struct ParallelZmapImpl;
        using EmbeddedIterator  = typename ZmapType::const_iterator;
        inline const_iterator(const ZmapType* inner, const ZmapType* inner_end,
                const EmbeddedIterator& it) :
            _inner(inner), _inner_end(inner_end), _it(it)  {
            if (inner) {
                _it_end = inner->end();
            }
        }

        inline void skip_empty() {
            while (_it == _it_end && _inner) {
                if (++_inner == _inner_end) {
                    _inner = nullptr; // marks end()
                    break;
                } else {
                    _it = _inner->begin();
                    _it_end = _inner->end();
                }
            }
        }

        const ZmapType* _inner      = nullptr;
        const ZmapType* _inner_end  = nullptr;
        EmbeddedIterator _it, _it_end;
    };

    inline const_iterator begin() const {
        const_iterator it = {&_subs[0], &_subs[_mask] + 1, _subs[0].begin()};
        it.skip_empty();
        return it;
    }
    inline const_iterator end() const { return const_iterator(); }

    inline std::pair<const_iterator, bool> emplace(KeyType key, const void* val,
            size_t val_len, size_t hash_val = -1) {
        if ((size_t)-1 == hash_val) hash_val = Hasher()(key);
        auto idx = subidx(hash_val);
        for (size_t i = 0, sc = _spilled_zmap_count[idx]; sc > 0 && i <= sc; ++i) {
            auto sub = &_subs[(idx + i) & _mask];
            auto it = sub->find(key, hash_val);
            if (ZMAP_PREDICT_TRUE(sub->end() != it)) {
                return {{sub, &_subs[_mask] + 1, it}, false};
            }
        }
        for (size_t i = 0; i <= _mask; ++i) {
            auto ret = _subs[(idx + i) & _mask].emplace(key, val, val_len, hash_val);
            if (ZMAP_PREDICT_FALSE(!ret.second && ret.first == _subs[(idx + i) & _mask].end())) {
                continue;
            }
            _spilled_zmap_count[idx] = std::max(_spilled_zmap_count[idx], i);
            return {{&_subs[(idx + i) & _mask], &_subs[_mask] + 1, ret.first}, ret.second};
        }
        return {end(), false};
    }
    inline std::pair<const_iterator, bool> emplace_s(KeyType key, const std::string& val,
            size_t hash_val = -1) {
        return emplace(key, (const void*)val.data(), val.size(), hash_val);
    }
    inline std::pair<const_iterator, bool> insert(const std::pair<RawKeyType, const void*>& kv,
            size_t hash_val = -1) {
        return emplace(kv.first, kv.second, hash_val);
    }
    inline std::pair<const_iterator, bool> insert(const std::pair<RawKeyType, std::string>& kv,
            size_t hash_val = -1) {
        return emplace(kv.first, kv.second.data(), kv.second.size(), hash_val);
    }

    inline int emplace_ex(KeyType key, const void* val, size_t val_len, size_t hash_val = -1) {
        if ((size_t)-1 == hash_val) hash_val = Hasher()(key);
        auto idx = subidx(hash_val);
        for (size_t i = 0, sc = _spilled_zmap_count[idx]; sc > 0 && i <= sc; ++i) {
            if (ZMAP_PREDICT_TRUE(_subs[(idx + i) & _mask].count(key, hash_val) > 0)) return 0;
        }
        for (size_t i = 0; i <= _mask; ++i) {
            auto ret = _subs[(idx + i) & _mask].emplace_ex(key, val, val_len, hash_val);
            if (ZMAP_PREDICT_FALSE(-1 == ret)) {
                continue;
            }
            _spilled_zmap_count[idx] = std::max(_spilled_zmap_count[idx], i);
            return ret;
        }
        return -1;
    }

    // ${return_val}.second is true if key doesn't exist before
    inline std::pair<const_iterator, bool> update(KeyType key, const void* val,
            size_t val_len, size_t hash_val = -1) {
        if ((size_t)-1 == hash_val) hash_val = Hasher()(key);
        auto idx = subidx(hash_val);
        for (size_t i = 0, sc = _spilled_zmap_count[idx]; sc > 0 && i <= sc; ++i) {
            auto sub = &_subs[(idx + i) & _mask];
            if (ZMAP_PREDICT_TRUE(sub->count(key, hash_val) > 0)) {
                auto ret = sub->update(key, val, val_len, hash_val);
                return {{&_subs[(idx + i) & _mask], &_subs[_mask] + 1, ret.first}, ret.second};
            }
        }
        for (size_t i = 0; i <= _mask; ++i) {
            auto ret = _subs[(idx + i) & _mask].update(key, val, val_len, hash_val);
            if (ZMAP_PREDICT_FALSE(!ret.second && ret.first == _subs[(idx + i) & _mask].end())) {
                continue;
            }
            _spilled_zmap_count[idx] = std::max(_spilled_zmap_count[idx], i);
            return {{&_subs[(idx + i) & _mask], &_subs[_mask] + 1, ret.first}, ret.second};
        }
        return {const_iterator(), false};
    }
    inline int update_ex(KeyType key, const void* val, size_t val_len, size_t hash_val = -1) {
        if ((size_t)-1 == hash_val) hash_val = Hasher()(key);
        auto idx = subidx(hash_val);
        for (size_t i = 0, sc = _spilled_zmap_count[idx]; sc > 0 && i <= sc; ++i) {
            auto sub = &_subs[(idx + i) & _mask];
            if (ZMAP_PREDICT_TRUE(sub->count(key, hash_val) > 0)) {
                return sub->update_ex(key, val, val_len, hash_val);
            }
        }
        for (size_t i = 0; i <= _mask; ++i) {
            auto ret = _subs[(idx + i) & _mask].update_ex(key, val, val_len, hash_val);
            if (ZMAP_PREDICT_FALSE(-1 == ret)) continue;
            _spilled_zmap_count[idx] = std::max(_spilled_zmap_count[idx], i);
            return ret;
        }
        return -1;
    }
    inline std::pair<const_iterator, bool> update_s(KeyType key, const std::string& val,
            size_t hash_val = -1) {
        return update(key, val.data(), val.size(), hash_val);
    }

    template <bool NO_DUP_KEY = false>
    inline int build(KeyType key, const void* val, size_t val_len = 0, size_t hash_val = -1) {
        if ((size_t)-1 == hash_val) hash_val = Hasher()(key);
        auto idx = subidx(hash_val);
        return _subs[idx & _mask].template build<NO_DUP_KEY>(key, val, val_len, hash_val);
    }

    inline size_t erase(KeyType key, size_t hash_val = -1) {
        if ((size_t)-1 == hash_val) hash_val = Hasher()(key);
        auto idx = subidx(hash_val);
        for (size_t i = 0, sc = _spilled_zmap_count[idx]; i <= sc; ++i) {
            if (ZMAP_PREDICT_TRUE(_subs[(idx + i) & _mask].erase(key, hash_val))) {
                return 1;
            }
        }
        return 0;
    }

    inline size_t erase(const const_iterator& it) {
        if (end() == it) return 0;
        return const_cast<ZmapType*>(it._inner)->erase(it._it);
    }

    inline const_iterator find(KeyType key) const {
        auto hash_val = Hasher()(key);
        auto idx = subidx(hash_val);
        for (size_t i = 0, sc = _spilled_zmap_count[idx]; i <= sc; ++i) {
            auto sub = &_subs[(idx + i) & _mask];
            auto it = sub->find(key, hash_val);
            if (ZMAP_PREDICT_TRUE(sub->end() != it)) {
                return {sub, &_subs[_mask] + 1, it};
            }
        }
        return const_iterator();
    }

    inline bool find_ex(KeyType key, const void*& val, size_t* val_len = nullptr) const {
        auto hash_val = Hasher()(key);
        auto idx = subidx(hash_val);
        for (size_t i = 0, sc = _spilled_zmap_count[idx]; i <= sc; ++i) {
            auto sub = &_subs[(idx + i) & _mask];
            if (ZMAP_PREDICT_TRUE(sub->find_ex(key, val, val_len, hash_val))) {
                return true;
            }
        }
        return false;
    }

    inline static size_t hash(KeyType key) { return Hasher()(key); }
    inline static size_t subidx(size_t hash_val) { return (hash_val ^ (hash_val >> N)) & _mask; }
    inline static size_t subcnt() { return _mask + 1; }

    inline size_t count(KeyType key) const {
        auto hash_val = Hasher()(key);
        auto idx = subidx(hash_val);
        for (size_t i = 0, sc = _spilled_zmap_count[idx]; i <= sc; ++i) {
            if (ZMAP_PREDICT_TRUE(_subs[(idx + i) & _mask].count(key, hash_val))) {
                return 1;
            }
        }
        return 0;
    }

    inline auto at(KeyType key) const {
        auto hash_val = Hasher()(key);
        auto idx = subidx(hash_val);
        const ZmapType* matched_sub = nullptr;
        for (size_t i = 0, sc = _spilled_zmap_count[idx]; i <= sc; ++i) {
            auto sub = &_subs[(idx + i) & _mask];
            if (ZMAP_PREDICT_TRUE(sub->count(key, hash_val))) {
                matched_sub = sub;
                break;
            }
        }
        if (!matched_sub) abort();
        return matched_sub->at(key);
    }

    // insert key with empty value if key doesn't exist in map,
    // but we couldn't return an reference of value
    inline auto operator[](KeyType key) {
        auto hash_val = Hasher()(key);
        auto idx = subidx(hash_val);
        for (size_t i = 0, sc = _spilled_zmap_count[idx]; sc > 0 && i <= sc; ++i) {
            auto sub = &_subs[(idx + i) & _mask];
            if (ZMAP_PREDICT_TRUE(sub->count(key, hash_val) > 0)) {
                return sub->operator[](key);
            }
        }
        for (size_t i = 0; i <= _mask; ++i) {
            auto sub = &_subs[(idx + i) & _mask];
            if (ZMAP_PREDICT_TRUE(-1 != sub->emplace_ex(key, nullptr, 0, hash_val))) {
                return sub->operator[](key);
            }
        }
        return decltype(std::declval<ZmapType>().operator[](0))();
    }

    // just facilitate for test
    inline RawKeyType rand_key() const {
        size_t idx = rand64() & _mask;
        auto key = _subs[idx].rand_key();
        if (subidx(Hasher()(key)) != idx) {
            for (size_t i = 0; i <= _mask; ++i) {
                for (const auto& x : _subs[(idx + i) & _mask]) {
                    if (subidx(Hasher()(x.first)) == ((idx + i) & _mask)) {
                        return x.first;
                    }
                }
            }
        }
        return key;
    }

    inline size_t size() const {
        size_t sz = 0;
        for (size_t i = 0; i <= _mask; ++i) {
            sz += _subs[i].size();
        }
        return sz;
    }

    inline size_t capacity() const {
        size_t capacity = 0;
        for (size_t i = 0; i <= _mask; ++i) {
            capacity += _subs[i].capacity();
        }
        return capacity;
    }

    inline bool empty() const {
        for (size_t i = 0; i <= _mask; ++i) {
            if (!_subs[i].empty()) {
                return false;
            }
        }
        return true;
    }

    inline void clear() {
        for (size_t i = 0; i <= _mask; ++i) {
            _subs[i].clear();
        }
        memset(_spilled_zmap_count, 0, sizeof(_spilled_zmap_count));
    }
    inline void reset() {
        for (size_t i = 0; i <= _mask; ++i) {
            _subs[i].reset();
        }
        memset(_spilled_zmap_count, 0, sizeof(_spilled_zmap_count));
    }

    inline size_t get_max_spilled_count() const {
        size_t ret = 0;
        for (size_t i = 0; i <= _mask; ++i) ret = std::max(ret, _spilled_zmap_count[i]);
        return ret;
    }

    // abort program if update failed when all sub maps are full, and
    // just set *catch_abort if it isn't nullptr.
    void batch_process(std::shared_ptr<KeyValueReader<RawKeyType>> reader,
            size_t threads_cnt = (1 << N), bool wait_done = false, bool* catch_abort = nullptr) {
        auto threads = std::make_shared<std::vector<std::thread>>();
        threads_cnt = std::min(threads_cnt, (1UL << N));
        struct Item {
            RawKeyType key;
            const void* val;
            size_t val_len;
        };

        bool has_spilled_items = false;
        for (size_t i = 0; i <= _mask; ++i) {
            if (_spilled_zmap_count[i] > 0) {
                has_spilled_items = true;
                break;
            }
        }

        static const size_t max_spilled_cnt = 5 * 1000 * 1000;
        auto total_spilled_cnt = std::make_shared<std::atomic<size_t>>(0);
        auto spilled_kvs = std::make_shared<std::vector<std::vector<Item>>>(threads_cnt);
        for (size_t thread_idx = 0; !has_spilled_items && thread_idx < threads_cnt; ++thread_idx) {
            threads->emplace_back([spilled_kvs, total_spilled_cnt, reader,
                                   threads_cnt, thread_idx, this]() {
                auto thread_reader = reader;
                if (thread_idx > 0) {
                    thread_reader.reset(reader->clone());
                }

                RawKeyType key{};
                const void* val = nullptr;
                size_t val_len = 0;
                bool erase_flag = false;

#ifndef NDEBUG
                auto tm_start = time(nullptr);
#endif
                while (thread_reader->next(key, val, &val_len, erase_flag)) {
                    auto hash_val = Hasher()(key);
                    auto idx = subidx(hash_val);
                    if (idx % threads_cnt != thread_idx) continue;

                    if (erase_flag) _subs[idx & _mask].erase(key);
                    else if (ZMAP_PREDICT_FALSE(-1 ==
                            _subs[idx & _mask].update_ex(key, val, val_len, hash_val)))
                    {
                        if (total_spilled_cnt->fetch_add(1, std::memory_order_acq_rel)
                                >= max_spilled_cnt) break;
                        (*spilled_kvs)[thread_idx].push_back({key, val, val_len});
                    }
                }
#ifndef NDEBUG
                printf("one thread finished for parallel zmap's batch_process, "
                       "thread_idx: %lu, now: %ld, spend: %ld\n", thread_idx,
                       time(nullptr), time(nullptr) - tm_start);
#endif
            });
        }

        std::thread process_spilled_kvs_thread([reader, threads, spilled_kvs, catch_abort,
                                                total_spilled_cnt, has_spilled_items, this]() {
            for (auto& thread : *threads) thread.join();

#ifndef NDEBUG
            auto tm_start = time(nullptr);
#endif
            if (has_spilled_items || total_spilled_cnt->load() >= max_spilled_cnt) {
                auto thread_reader = reader;
                thread_reader.reset(reader->clone());
                RawKeyType key{};
                const void* val = nullptr;
                size_t val_len = 0;
                bool erase_flag = false;
                while (thread_reader->next(key, val, &val_len, erase_flag)) {
                    auto hash_val = Hasher()(key);
                    if (erase_flag) erase(key);
                    else if (ZMAP_PREDICT_FALSE(-1 == update_ex(key, val, val_len, hash_val))) {
                        zmap_perror("map is full, please increase capacity when init");
                        if (catch_abort) *catch_abort = true;
                        else abort();
                    }
                }
            } else {
                for (const auto& th : *spilled_kvs) {
                    for (const auto& item : th) {
                        if (ZMAP_PREDICT_FALSE(-1 ==
                                update_ex(item.key, item.val, item.val_len))) {
                            zmap_perror("map is full, please increase capacity when init");
                            if (catch_abort) *catch_abort = true;
                            else abort();
                        }
                    }
                }
            }
#ifndef NDEBUG
            if (total_spilled_cnt->load() > 0) {
                printf("the thread that processes spilled items finished for parallel zmap's "
                       "batch_process, total spilled cnt:%lu, process whole data or not: %d, "
                       "now: %ld, spend: %ld seconds\n", total_spilled_cnt->load(),
                       has_spilled_items || total_spilled_cnt->load() >= max_spilled_cnt,
                       time(nullptr), time(nullptr) - tm_start);
            }
#endif
        });

        wait_done ? process_spilled_kvs_thread.join() : process_spilled_kvs_thread.detach();
    }

    void debug(std::ostream* out = nullptr, size_t sub_idx = -1) const {
        std::ostringstream oss;
        std::ostream* pout = out ? out : &oss;
        if ((size_t)-1 != sub_idx) {
            *pout << string_printf("zmap[%lu], size: %lu, capacity: %lu, deleted: %lu, "
                    "max_size: %lu, left: %lu, spilled_zmap_count: 0\n", sub_idx,
                    _subs[sub_idx].size(), _subs[sub_idx].header().capacity,
                    _subs[sub_idx].header().deleted, _subs[sub_idx].capacity(),
                    _subs[sub_idx].capacity() - _subs[sub_idx].size());
        } else {
            *pout << "[parallel zmap]\n";
            size_t total_sz = 0;
            size_t total_capacity = 0;
            size_t total_deleted = 0;
            size_t total_left = 0;
            size_t total_max_size = 0;
            for (size_t i = 0; i <= _mask; ++i) {
                auto cur_sz = _subs[i].size();
                auto cur_capacity = _subs[i].header().capacity;
                auto cur_max_size = _subs[i].capacity();
                auto cur_left = cur_max_size - cur_sz;
                auto cur_deleted = _subs[i].header().deleted;
                total_sz += cur_sz;
                total_capacity += cur_capacity;
                total_deleted += cur_deleted;
                total_left += cur_left;
                total_max_size += cur_max_size;
                *pout << string_printf("zmap[%lu], size: %lu, capacity: %lu, deleted: %lu, "
                        "max_size: %lu, left: %lu, spilled_zmap_count: %lu\n", i, cur_sz,
                        cur_capacity, cur_deleted, cur_max_size, cur_left, _spilled_zmap_count[i]);
            }
            *pout << string_printf("zmap total size: %lu, total capacity: %lu, max load factor: %f, "
                    "max size: %lu, total deleted: %lu, total left: %lu\n", total_sz,
                    total_capacity, MAX_LOAD_FACTOR, total_max_size, total_deleted, total_left);
        }
        if (!out) printf("%s", oss.str().data());
    }

private:
    ZmapType _subs[1 << N];
    size_t _spilled_zmap_count[1 << N];
    constexpr static size_t _mask = (1 << N) - 1;
};

} // end of namespace internal

#if defined(__AVX512F__) && defined(__AVX512BW__) && __GNUC__ >= 6
#warning Tip: you are using ProbeGroup512 to probe, please make sure deployed machines \
         support avx512f and avx512bw
    template <size_t ValueLength, typename Hasher = DefaultHasher>
    using Zmap = internal::ZmapImpl<internal::ProbeGroup512, ValueLength, Hasher>;

    template <typename Hasher = DefaultHasher>
    using ElasticZmapEx = internal::ElasticZmapImpl<internal::ProbeGroup512, Hasher>;

    using StringZmap = internal::StringZmapImpl<internal::ProbeGroup512>;
#elif defined(__AVX__) && defined(__AVX2__)
#warning Tip: you are using ProbeGroup256 to probe, please make sure deployed machines \
         support avx2
    template <size_t ValueLength, typename Hasher = DefaultHasher>
    using Zmap = internal::ZmapImpl<internal::ProbeGroup256, ValueLength, Hasher>;

    template <typename Hasher = DefaultHasher>
    using ElasticZmapEx = internal::ElasticZmapImpl<internal::ProbeGroup256, Hasher>;

    using StringZmap = internal::StringZmapImpl<internal::ProbeGroup256>;
#else
# warning Tip: you are using ProbeGroup128 to probe, try to compile with -mavx2 for a \
          better performance even if under poor condition
    template <size_t ValueLength, typename Hasher = DefaultHasher>
    using Zmap = internal::ZmapImpl<internal::ProbeGroup128, ValueLength, Hasher>;

    template <typename Hasher = DefaultHasher>
    using ElasticZmapEx = internal::ElasticZmapImpl<internal::ProbeGroup128, Hasher>;

    using StringZmap = internal::StringZmapImpl<internal::ProbeGroup128>;
#endif

// For convenience, expose it here once you need an allocator that supports dump/load via disk.
// Note that it only support single thread to do the allocation.
using ZmapAlloc = internal::DefaultAlloc;

using ElasticZmap = ElasticZmapEx<>;

template <size_t ValueLength, size_t N = 4, typename Hasher = DefaultHasher>
using ParallelZmap = internal::ParallelZmapImpl<Zmap<ValueLength, Hasher>, N>;

template <size_t N = 4, typename Hasher = DefaultHasher>
using ParallelElasticZmapEx = internal::ParallelZmapImpl<ElasticZmapEx<Hasher>, N>;
using ParallelElasticZmap = ParallelElasticZmapEx<>;

template <size_t N = 4>
using ParallelStringZmapEx = internal::ParallelZmapImpl<StringZmap, N>;
using ParallelStringZmap = ParallelStringZmapEx<>;

using Zset = Zmap<0>;
using StringZset = StringZmap;

///////////////////////////////////////
// Recap the usage of zmap
///////////////////////////////////////
// 1. Basic types:
//    Zmap<ValueLength>         // key: uint64_t, val: fixed length
//    ElasticZmap               // key: uint64_t, val: string
//    StringZmap                // key: string,   val: string
//
// 2. Parallel types:
//    ParallelZmap<ValueLength> // key: uint64_t, val: fixed length
//    ParallelElasticZmap       // key: uint64_t, val: string
//    ParallelStringZmap        // key: string,   val: string
//
// 3. Could it be used as set? Yes
//    Zmap<0>                   // key: uint64_t
//    StringZmap                // key: string
//    please remember to emplace with val==nullptr and val_len==0
//
// 4. Could an user-defined object use zmap? Yes
//    first of all, the object should support a method to generate one signature that represents
//    itself uniquely, and use this signature as the key for zmap; secondly, the object could be
//    serialized into a string as the value for zmap; finally, choose the right zmap type based
//    on the type of signature: ElasticZmap or StringZmap.

} // end of namespace zmap

#endif /* _ZMAP_H_ */
