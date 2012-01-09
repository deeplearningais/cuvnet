#include <algorithm>
#include <boost/filesystem.hpp>
#include <boost/lexical_cast.hpp>
#include "data_source.hpp"

using namespace cuvnet;
namespace fs = boost::filesystem;

folder_loader::folder_loader(const std::string& pathstr, bool recursive){
    scan(pathstr, recursive);
}
void
folder_loader::scan(const std::string& pathstr, bool recursive){
    m_files.clear();
    fs::path folder(pathstr);

    std::vector<std::string> allowed_extensions;
    allowed_extensions.push_back("jpg");
    allowed_extensions.push_back("JPG");
    allowed_extensions.push_back("jpeg");
    allowed_extensions.push_back("JPEG");

    for(fs::directory_iterator it = fs::directory_iterator(folder);
            it != fs::directory_iterator(); it++){
        if(std::find(allowed_extensions.begin(),
                    allowed_extensions.end(),
                    fs::extension(*it))
                != allowed_extensions.end()){
            m_files.push_back(detail::file_descr());
            m_files.back().name = boost::lexical_cast<std::string>(*it);
            m_files.back().size = fs::file_size(*it);
        }
    }
}
