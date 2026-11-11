#include "benchmark_impl.hpp"

int main(int argc, char** argv) {
    try {
        setbuf(stdout, nullptr); Options o = options(argc, argv);
        uint16_t endian = 1; check(*reinterpret_cast<uint8_t*>(&endian) == 1, "little-endian required");
        if(!o.dump_indices.empty()){
            for(int epoch=1;epoch<=o.rounds+o.warmups;++epoch){
                auto raw=generate_indices(o,epoch);
                std::string path=o.dump_indices+"/indices-r"+std::to_string(o.rank)+"-t"+std::to_string(o.groups)+"-"+o.distribution+"-e"+std::to_string(epoch)+".bin";
                std::ofstream f(path,std::ios::binary);f.write(reinterpret_cast<const char*>(raw.data()),raw.size()*4);check(f.good(),"write raw upstream indices");
            }
            return 0;
        }
        if (o.planner_profile) run_planner_profile(o);
        else if (o.owner) run_owner(o); else run_requester(o);
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "ERROR: %s\n", e.what()); return 1;
    }
}
