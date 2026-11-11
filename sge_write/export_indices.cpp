#include "input.hpp"
int main(int argc,char**argv){
    try{Options o=options(argc,argv);check(!o.dump_indices.empty(),"--dump-indices required");uint16_t endian=1;check(*reinterpret_cast<uint8_t*>(&endian)==1,"little-endian input format");
        for(int epoch=1;epoch<=o.rounds+o.warmups;++epoch){auto raw=generate_indices(o,epoch);std::string p=o.dump_indices+"/indices-r"+std::to_string(o.rank)+"-t"+std::to_string(o.groups)+"-"+o.distribution+"-e"+std::to_string(epoch)+".bin";
            std::ofstream f(p,std::ios::binary);f.write(reinterpret_cast<const char*>(raw.data()),raw.size()*4);check(f.good(),"write indices");}
        return 0;
    }catch(const std::exception&e){std::fprintf(stderr,"input export: %s\n",e.what());return 1;}
}
