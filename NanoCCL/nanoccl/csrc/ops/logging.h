// NanoCCL logging - C++17 compatible (no std::source_location)
#pragma once

#include <cstdlib>
#include <iostream>
#include <mutex>
#include <sstream>
#include <string>

namespace nanoccl {

inline std::string get_env_variable(char const* env_var_name)
{
    if (!env_var_name) {
        return "";
    }
    char* lvl = getenv(env_var_name);
    if (lvl)
        return std::string(lvl);
    return "";
}

inline int& get_log_level_internal()
{
    static int level = []() {
        std::string lvl_str = get_env_variable("NANOCCL_LOG_LEVEL");
        if (lvl_str.empty())
            return 1;  // Default to INFO (1). 0=ERROR, 1=INFO, 2=DEBUG
        return std::stoi(lvl_str);
    }();
    return level;
}

inline int get_log_level()
{
    return get_log_level_internal();
}

inline bool is_mutex_logging_enabled()
{
    static bool enabled = []() {
        std::string val = get_env_variable("NANOCCL_LOG_MUTEX");
        return !val.empty() && std::stoi(val) != 0;
    }();
    return enabled;
}

inline std::mutex& get_console_mutex()
{
    static std::mutex mtx;
    return mtx;
}

#define NANOCCL_CONSOLE_LOCK                                                                                           \
    std::unique_lock<std::mutex> _nanoccl_console_lock(nanoccl::get_console_mutex(), std::defer_lock);                 \
    if (nanoccl::is_mutex_logging_enabled()) {                                                                         \
        _nanoccl_console_lock.lock();                                                                                  \
    }

template<typename... Args>
void log_message_impl(
    int level, const char* level_str, const char* color, const char* file, int line, const char* func, Args&&... args)
{
    if (get_log_level() >= level) {
        NANOCCL_CONSOLE_LOCK
        std::cerr << color << "[" << level_str << "]\033[m " << file << ":" << line << ": " << func << ": ";
        ((std::cerr << std::forward<Args>(args)), ...);
        std::cerr << std::endl;
    }
}

template<typename... Args>
void assertion_failed_impl(const char* expr_str, const char* file, int line, const char* func, Args&&... args)
{
    {
        NANOCCL_CONSOLE_LOCK
        std::cerr << "\033[1;91m[Assertion Failed]\033[m " << file << ":" << line << ": " << func
                  << ", Expected: " << expr_str << ". Error msg: ";
        ((std::cerr << std::forward<Args>(args)), ...);
        std::cerr << std::endl;
    }
    abort();
}

#define NANOCCL_LOG_ERROR(...)                                                                                         \
    nanoccl::log_message_impl(0, "ERROR", "\033[1;91m", __FILE__, __LINE__, __FUNCTION__, __VA_ARGS__)
#define NANOCCL_LOG_WARN(...)                                                                                          \
    nanoccl::log_message_impl(1, "WARN", "\033[1;93m", __FILE__, __LINE__, __FUNCTION__, __VA_ARGS__)
#define NANOCCL_LOG_INFO(...)                                                                                          \
    nanoccl::log_message_impl(1, "INFO", "\033[1;92m", __FILE__, __LINE__, __FUNCTION__, __VA_ARGS__)
#define NANOCCL_LOG_DEBUG(...)                                                                                         \
    nanoccl::log_message_impl(2, "DEBUG", "\033[1;94m", __FILE__, __LINE__, __FUNCTION__, __VA_ARGS__)

#define NANOCCL_ASSERT(Expr, ...)                                                                                      \
    if (!(Expr)) {                                                                                                     \
        nanoccl::assertion_failed_impl(#Expr, __FILE__, __LINE__, __FUNCTION__ __VA_OPT__(, ) __VA_ARGS__);            \
    }

#define NANOCCL_ASSERT_EQ(A, B, ...) NANOCCL_ASSERT((A) == (B), __VA_ARGS__)
#define NANOCCL_ASSERT_NE(A, B, ...) NANOCCL_ASSERT((A) != (B), __VA_ARGS__)
#define NANOCCL_ASSERT_GT(A, B, ...) NANOCCL_ASSERT((A) > (B), __VA_ARGS__)
#define NANOCCL_ASSERT_GE(A, B, ...) NANOCCL_ASSERT((A) >= (B), __VA_ARGS__)
#define NANOCCL_ASSERT_LT(A, B, ...) NANOCCL_ASSERT((A) < (B), __VA_ARGS__)
#define NANOCCL_ASSERT_LE(A, B, ...) NANOCCL_ASSERT((A) <= (B), __VA_ARGS__)

}  // namespace nanoccl
