//
// Created by jimy on 3/13/22.
//

#pragma once

#include <cxxabi.h>
#include <execinfo.h>
#include <unistd.h>

#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

namespace nanodeploy {

// -----------------------------------------------------------------------------
// Stack Trace Utility
// -----------------------------------------------------------------------------

/**
 * @brief Captures and prints the current stack trace to stderr.
 *
 * This function attempts to demangle C++ function names for readability.
 * It uses ANSI colors to highlight the output, consistent with the logging macros.
 *
 * Note: For function names to appear, compile with "-rdynamic" (GCC/Clang).
 * For debug info (line numbers), compile with "-g".
 */
inline void print_stack_trace()
{
    const int max_frames = 64;
    void*     addr_list[max_frames];

    // Retrieve current stack addresses
    int addr_len = backtrace(addr_list, max_frames);

    if (addr_len == 0) {
        std::cerr << "  <empty, possibly corrupt stack>" << std::endl;
        return;
    }

    // Resolve addresses into strings
    char** symbol_list = backtrace_symbols(addr_list, addr_len);
    if (!symbol_list) {
        std::cerr << "  <failed to resolve stack symbols>" << std::endl;
        return;
    }

    // Wrap the pointer to ensure it gets freed automatically
    std::unique_ptr<char*, void (*)(void*)> symbol_guard(symbol_list, [](void* p) { free(p); });

    std::cerr << "\033[1;96m"
              << "\n--- Stack Trace ---\n"
              << "\033[m";

    for (int i = 0; i < addr_len; ++i) {
        std::string symbol = symbol_list[i];
        std::string func_name;
        std::string offset;
        std::string address;

        // Attempt to parse the symbol string.
        // Format is usually: ./executable(function_name+0x123) [0x456]
        // This varies by OS/Compiler, so we do a best-effort parse.
        size_t open_paren  = symbol.find('(');
        size_t plus_sign   = symbol.find('+', open_paren);
        size_t close_paren = symbol.find(')', plus_sign);

        if (open_paren != std::string::npos && plus_sign != std::string::npos && close_paren != std::string::npos) {

            std::string mangled = symbol.substr(open_paren + 1, plus_sign - open_paren - 1);
            offset              = symbol.substr(plus_sign, close_paren - plus_sign);
            address             = symbol.substr(close_paren + 1);

            // Demangle the name
            int                                    status = -1;
            std::unique_ptr<char, void (*)(void*)> demangled(
                abi::__cxa_demangle(mangled.c_str(), nullptr, nullptr, &status), [](void* p) { std::free(p); });

            if (status == 0 && demangled) {
                func_name = demangled.get();
            }
            else {
                func_name = mangled;  // Fallback to mangled name
            }
        }
        else {
            // Could not parse, just print the raw line
            func_name = symbol;
        }

        // Print formatted frame
        std::cerr << "#" << i << " " << func_name << " "
                  << "\033[90m" << offset << " " << address << "\033[m" << std::endl;
    }
    std::cerr << std::endl;
}

// -----------------------------------------------------------------------------
// Macro Helpers (Variadic Argument Handling)
// -----------------------------------------------------------------------------

#define STREAM_VAR_ARGS1(a) << a
#define STREAM_VAR_ARGS2(a, b) << a << b
#define STREAM_VAR_ARGS3(a, b, c) << a << b << c
#define STREAM_VAR_ARGS4(a, b, c, d) << a << b << c << d
#define STREAM_VAR_ARGS5(a, b, c, d, e) << a << b << c << d << e
#define STREAM_VAR_ARGS6(a, b, c, d, e, f) << a << b << c << d << e << f
#define STREAM_VAR_ARGS7(a, b, c, d, e, f, g) << a << b << c << d << e << f << g
#define STREAM_VAR_ARGS8(a, b, c, d, e, f, g, h) << a << b << c << d << e << f << g << h

#define GET_MACRO(_1, _2, _3, _4, _5, _6, _7, _8, NAME, ...) NAME

#define STREAM_VAR_ARGS(...)                                                                                           \
    GET_MACRO(__VA_ARGS__,                                                                                             \
              STREAM_VAR_ARGS8,                                                                                        \
              STREAM_VAR_ARGS7,                                                                                        \
              STREAM_VAR_ARGS6,                                                                                        \
              STREAM_VAR_ARGS5,                                                                                        \
              STREAM_VAR_ARGS4,                                                                                        \
              STREAM_VAR_ARGS3,                                                                                        \
              STREAM_VAR_ARGS2,                                                                                        \
              STREAM_VAR_ARGS1)                                                                                        \
    (__VA_ARGS__)

// -----------------------------------------------------------------------------
// Assertions
// -----------------------------------------------------------------------------

#define NANODEPLOY_ASSERT(Expr, Msg, ...)                                                                              \
    {                                                                                                                  \
        if (!(Expr)) {                                                                                                 \
            NANODEPLOY_CONSOLE_LOCK                                                                                    \
            std::cerr << "\033[1;91m"                                                                                  \
                      << "[Assertion Failed]"                                                                          \
                      << "\033[m " << __FILE__ << ":" << __LINE__ << ": " << __FUNCTION__ << ", Expected: " << #Expr   \
                      << ". Error msg: " << Msg __VA_OPT__(STREAM_VAR_ARGS(__VA_ARGS__)) << std::endl;                 \
            nanodeploy::print_stack_trace(); /* Dump stack before aborting */                                          \
            abort();                                                                                                   \
        }                                                                                                              \
    }

#define NANODEPLOY_ASSERT_EQ(A, B, Msg, ...) NANODEPLOY_ASSERT((A) == (B), Msg, __VA_ARGS__)
#define NANODEPLOY_ASSERT_NE(A, B, Msg, ...) NANODEPLOY_ASSERT((A) != (B), Msg, __VA_ARGS__)
#define NANODEPLOY_ASSERT_GT(A, B, Msg, ...) NANODEPLOY_ASSERT((A) > (B), Msg, __VA_ARGS__)
#define NANODEPLOY_ASSERT_GE(A, B, Msg, ...) NANODEPLOY_ASSERT((A) >= (B), Msg, __VA_ARGS__)
#define NANODEPLOY_ASSERT_LT(A, B, Msg, ...) NANODEPLOY_ASSERT((A) < (B), Msg, __VA_ARGS__)
#define NANODEPLOY_ASSERT_LE(A, B, Msg, ...) NANODEPLOY_ASSERT((A) <= (B), Msg, __VA_ARGS__)

#define NANODEPLOY_ABORT(Msg, ...)                                                                                     \
    {                                                                                                                  \
        std::cerr << "\033[1;91m"                                                                                      \
                  << "[Fatal]"                                                                                         \
                  << "\033[m " << __FILE__ << ":" << __LINE__ << ": " << __FUNCTION__ << ": "                          \
                  << Msg __VA_OPT__(STREAM_VAR_ARGS(__VA_ARGS__)) << std::endl;                                        \
        nanodeploy::print_stack_trace();                                                                               \
        abort();                                                                                                       \
    }

// -----------------------------------------------------------------------------
// Logging
// -----------------------------------------------------------------------------

#define NANODEPLOY_LOG_LEVEL(MsgType, FlagFormat, Level, ...)                                                          \
    {                                                                                                                  \
        if (get_log_level() >= Level) {                                                                                \
            std::cerr << FlagFormat << "[" << MsgType << "]"                                                           \
                      << "\033[m " << __FILE__ << ":" << __LINE__ << ": " << __FUNCTION__                              \
                      << ": " __VA_OPT__(STREAM_VAR_ARGS(__VA_ARGS__)) << std::endl;                                   \
        }                                                                                                              \
    }

// Error and Warn
#define NANODEPLOY_LOG_ERROR(...) NANODEPLOY_LOG_LEVEL("ERROR", "\033[1;91m", 0, __VA_ARGS__)
#define NANODEPLOY_LOG_WARN(...) NANODEPLOY_LOG_LEVEL("WARN", "\033[1;91m", 1, __VA_ARGS__)

// Info
#define NANODEPLOY_LOG_INFO(...) NANODEPLOY_LOG_LEVEL("INFO", "\033[1;92m", 1, __VA_ARGS__)

// Debug
#define NANODEPLOY_LOG_DEBUG(...) NANODEPLOY_LOG_LEVEL("DEBUG", "\033[1;92m", 2, __VA_ARGS__)

}  // namespace nanodeploy
