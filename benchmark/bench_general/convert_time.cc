#include <stdio.h>
#include <string.h>
#include <math.h>
#include <numeric>
#include <iostream>
#include <fstream>
#include <chrono>
#include <thread>
#include <fstream>
#include <iterator>
#include <vector>
#include <algorithm>
#include <iostream>
#include <string>
#include <sstream>

template <typename T>
static void 
process_dat(std::string filename) {

    std::fstream file(filename);
    
    int offset = 0;
    int line_length = 10;
    int numlines = 7;
    std::vector<std::string> all_data(numlines * line_length);

    for( std::string l; getline(file, l);)
    {
        std::stringstream ss(l);
        std::istream_iterator<std::string> begin(ss);
        std::istream_iterator<std::string> end;
        std::vector<std::string> times_per_col_sz(begin, end);

        
        std::for_each(times_per_col_sz.begin(), times_per_col_sz.end(),
            [](std::string& entry) {
                entry = std::to_string((std::stod(entry) / 1e6));
            }
        );

        std::copy(times_per_col_sz.begin(), times_per_col_sz.end(), all_data.begin() + offset);
        offset += line_length;
    }
    
    // Clear file
    std::ofstream ofs;
    ofs.open(filename, std::ofstream::out | std::ofstream::trunc);   
    // re-open file
    std::fstream file1(filename, std::fstream::app);
 
    for(int i = 0; i < line_length * numlines; ++i)
    {
        if(!(i % line_length))
            file1 << "\n";
        file1 << all_data[i] << "  ";
    }
}

int main(){ 
    process_dat<double>("speed/raw_data/embedding_combined.dat");
    return 0;
}