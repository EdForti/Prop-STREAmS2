#include <map>
#include <string>
#include <iostream>
#include <typeinfo>

#include "cantera/core.h"
#include "cantera/thermo.h"
#include "cantera/base/AnyMap.h"
#include "cantera/base/global.h"
#include "cantera/thermo/Nasa9PolyMultiTempRegion.h"

using namespace Cantera;

extern "C" {

void get_cp_from_cantera_c(int i_specie, int *n_ranges_ret, int *n_coeffs_ret, double **tem_ranges, double **cp_coeff_matrix) {

	  int n_coeffs = 9;
		string infile = "input_cantera.yaml";
		auto rootNode = AnyMap::fromYamlFile(infile);
//    auto sol    = newSolution(infile);
//    auto gas    = sol->thermo();
  
		vector<AnyMap> species                     = rootNode["species"].asVector<AnyMap>();
		auto &specie                               = species[i_specie-1];  // consider F90 lbound 1
		vector<double> tem_ranges_vec              = specie["thermo"]["temperature-ranges"].asVector<double>();
		vector<vector<double>> cp_coeff_ranges_vec = specie["thermo"]["data"].asVector<vector<double>>();

		int n_ranges = tem_ranges_vec.size()-1;
		//std::cout << "n_ranges from cxx: " << n_ranges << std::endl;
		//double *tem_ranges_arr = &tem_ranges[0];
		*tem_ranges = (double*) malloc((n_ranges+1) * sizeof(double));
		*cp_coeff_matrix = (double*) malloc(n_coeffs * n_ranges * sizeof(double));
    for(unsigned int i = 0; i < n_ranges+1; i++) {
        (*tem_ranges)[i] = tem_ranges_vec[i];
				//std::cout << "tem_ranges[i]: " << (*tem_ranges)[i] << std::endl;
		}
    for(unsigned int i = 0; i < n_ranges; i++) {
        for(unsigned int j = 0; j < n_coeffs; j++) {
            (*cp_coeff_matrix)[j+i*n_coeffs] = cp_coeff_ranges_vec[i][j];
				}
		}

		*n_ranges_ret = n_ranges;
		*n_coeffs_ret = n_coeffs;

}

}
