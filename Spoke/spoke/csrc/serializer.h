#pragma once
#include <cstring>
#include <iostream>
#include <string>
#include <type_traits>
#include <vector>

namespace spoke {

// 1. 定义序列化接口模板 (默认实现：二进制 memcpy)
// 1. 定义序列化接口模板 (默认实现：二进制 memcpy)
template<typename T, typename Enable = void>
struct Serializer {
    static std::string pack(const T& obj)
    {
        std::string s;
        s.resize(sizeof(T));
        std::memcpy(s.data(), &obj, sizeof(T));
        return s;
    }

    static T unpack(const std::string& data)
    {
        T obj;
        if (data.size() >= sizeof(T)) {
            std::memcpy(&obj, data.data(), sizeof(T));
        }
        return obj;
    }

    // [New] Direct Buffer API
    static size_t size(const T& obj)
    {
        return sizeof(T);
    }

    static void packTo(const T& obj, char* buf)
    {
        std::memcpy(buf, &obj, sizeof(T));
    }

    static T unpackFrom(const char* buf, size_t len)
    {
        T obj;
        if (len >= sizeof(T)) {
            std::memcpy(&obj, buf, sizeof(T));
        }
        return obj;
    }
};

// 2. 特化：针对 std::string
template<>
struct Serializer<std::string> {
    static std::string pack(const std::string& obj)
    {
        return obj;
    }
    static std::string unpack(const std::string& data)
    {
        return data;
    }

    static size_t size(const std::string& obj)
    {
        return obj.size();
    }

    static void packTo(const std::string& obj, char* buf)
    {
        std::memcpy(buf, obj.data(), obj.size());
    }

    static std::string unpackFrom(const char* buf, size_t len)
    {
        return std::string(buf, len);
    }
};

// 3. 特化：针对 std::vector (基本类型) - only for trivially copyable types
template<typename T>
struct Serializer<std::vector<T>, std::enable_if_t<std::is_trivially_copyable<T>::value>> {
    static std::string pack(const std::vector<T>& obj)
    {
        std::string s;
        size_t      bytes = obj.size() * sizeof(T);
        s.resize(bytes);
        std::memcpy(s.data(), obj.data(), bytes);
        return s;
    }

    static std::vector<T> unpack(const std::string& data)
    {
        size_t         count = data.size() / sizeof(T);
        std::vector<T> vec(count);
        std::memcpy(vec.data(), data.data(), data.size());
        return vec;
    }

    static size_t size(const std::vector<T>& obj)
    {
        return obj.size() * sizeof(T);
    }

    static void packTo(const std::vector<T>& obj, char* buf)
    {
        std::memcpy(buf, obj.data(), obj.size() * sizeof(T));
    }

    static std::vector<T> unpackFrom(const char* buf, size_t len)
    {
        size_t         count = len / sizeof(T);
        std::vector<T> vec(count);
        std::memcpy(vec.data(), buf, len);
        return vec;
    }
};

// 4. 辅助函数
template<typename T>
std::string Pack(const T& obj)
{
    return Serializer<T>::pack(obj);
}

template<typename T>
T Unpack(const std::string& raw)
{
    return Serializer<T>::unpack(raw);
}

// [New] Direct Buffer Helpers
template<typename T>
size_t PackedSize(const T& obj)
{
    return Serializer<T>::size(obj);
}

template<typename T>
void PackTo(const T& obj, char* buf)
{
    Serializer<T>::packTo(obj, buf);
}

template<typename T>
T UnpackFrom(const char* buf, size_t len)
{
    return Serializer<T>::unpackFrom(buf, len);
}

}  // namespace spoke
