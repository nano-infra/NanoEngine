#include <sstream>
#include <vector>

#include "spoke/csrc/actor.h"

struct ComplexTask {
    int                 id;
    std::string         name;
    std::vector<double> values;
};

namespace spoke {
template<>
struct Serializer<ComplexTask> {
    static std::string pack(const ComplexTask& t)
    {
        std::stringstream ss;
        ss << t.id << "|" << t.name << "|";
        for (double v : t.values)
            ss << v << ",";
        return ss.str();
    }
    static ComplexTask unpack(const std::string& s)
    {
        ComplexTask       t;
        std::stringstream ss(s);
        std::string       seg;

        std::getline(ss, seg, '|');
        t.id = std::stoi(seg);
        std::getline(ss, t.name, '|');
        std::string vals;
        std::getline(ss, vals);
        std::stringstream vs(vals);
        std::string       v;
        while (std::getline(vs, v, ','))
            t.values.push_back(std::stod(v));
        return t;
    }

    // [New] Zero-Copy Interface
    static size_t size(const ComplexTask& t)
    {
        return pack(t).size();
    }

    static void packTo(const ComplexTask& t, char* buf)
    {
        std::string s = pack(t);
        memcpy(buf, s.data(), s.size());
    }

    static ComplexTask unpackFrom(const char* buf, size_t size)
    {
        return unpack(std::string(buf, size));
    }
};
}  // namespace spoke

const spoke::Action kProcessTask = (spoke::Action)0x60;

class AdvancedActor: public spoke::Actor {
public:
    using spoke::Actor::Actor;

    SPOKE_METHOD(AdvancedActor, process, kProcessTask, ComplexTask, std::string)
    {
        double sum = 0;
        for (auto v : val.values)
            sum += v;

        std::string res = "Processed " + val.name + " (ID:" + std::to_string(val.id) + ") Sum=" + std::to_string(sum);
        return res;
    }
};

SPOKE_REGISTER_ACTOR("AdvancedActor", AdvancedActor);
