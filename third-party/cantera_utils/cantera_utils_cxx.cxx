#include <map>
#include <string>
#include <iostream>
#include <typeinfo>
#include <vector>

#include "cantera/core.h"
#include "cantera/kinetics/Arrhenius.h"
#include "cantera/kinetics/Reaction.h"
#include "cantera/kinetics/ReactionData.h"
#include "cantera/kinetics/ThirdBodyCalc.h"
#include "cantera/kinetics/Falloff.h"
#include "cantera/thermo.h"
#include "cantera/thermo/IdealGasPhase.h"
#include "cantera/base/AnyMap.h"
#include "cantera/base/global.h"
#include "cantera/thermo/Nasa9PolyMultiTempRegion.h"
#include "cantera/thermo/NasaPoly2.h"
#include "cantera/thermo/NasaPoly1.h"
#include "cantera/base/Solution.h"
#include "cantera/clib/ct.h"
#include "cantera/transport.h"
#include "cantera/transport/GasTransport.h"
#include "cantera/transport/TransportData.h"

using namespace Cantera;

extern "C" {

void get_transport_data_from_cantera_c(int* n_species_ret, double** sigma, double** epsK, double** dipole, double** polariz, int** geometry)
{
    auto sol = newSolution("input_cantera.yaml");
    auto th  = sol->thermo();
    int nsp  = th->nSpecies();

    *n_species_ret = nsp;

    *sigma    = (double*) malloc(nsp * sizeof(double));
    *epsK     = (double*) malloc(nsp * sizeof(double));
    *dipole   = (double*) malloc(nsp * sizeof(double));
    *polariz  = (double*) malloc(nsp * sizeof(double));
    *geometry = (int*)    malloc(nsp * sizeof(int));

    for (int lsp = 0; lsp < nsp; ++lsp) {
        auto sp  = th->species(lsp);
        auto trdata = sp->transport;
        auto gasdata = std::dynamic_pointer_cast<GasTransportData>(trdata);
        if (!gasdata) throw std::runtime_error("Species transport is not GasTransportData type");

        (*sigma)[lsp]   = gasdata->diameter;
        (*epsK)[lsp]    = gasdata->well_depth / Boltzmann;
        (*dipole)[lsp]  = gasdata->dipole;
        (*polariz)[lsp] = gasdata->polarizability;
        
        if (gasdata->geometry == "atom"){
            (*geometry)[lsp] = 0;
        } else if (gasdata->geometry == "linear"){
            (*geometry)[lsp] = 1;
        } else if (gasdata->geometry == "nonlinear"){
            (*geometry)[lsp] = 2;
        } else {
            throw std::runtime_error(
                "Unknown geometry string for species '" + sp->name + "': " + gasdata->geometry
            );
        }
    }
}

void get_cp_from_cantera_c(int i_specie, int *n_ranges_ret, int *n_coeffs_ret, double **temp_ranges, double **cp_coeff_matrix) {

      //auto sol     = newSolution("air.yaml");
      auto sol     = newSolution("input_cantera.yaml");
      auto th      = sol->thermo();
      int  nsp     = th->nSpecies();

      i_specie = i_specie - 1;
      std::string speciesName = th->speciesName(i_specie);
      auto  sp = th->species(i_specie);

      auto  nasa7poly1 = dynamic_cast<const Cantera::NasaPoly1*>(sp->thermo.get());
      auto  nasa7poly2 = dynamic_cast<const Cantera::NasaPoly2*>(sp->thermo.get());
      auto  nasa9poly  = dynamic_cast<const Cantera::Nasa9PolyMultiTempRegion*>(sp->thermo.get());

      if (nasa9poly) {
       int n_coeffs = 9;
       size_t species_index;
       int type;
       double tlow, thigh, pref;
       double* coeffs = static_cast<double*>(malloc((1 + 11 * 10)*sizeof(double))); // Assuming a maximum of 10 zones

       nasa9poly->reportParameters(species_index, type, tlow, thigh, pref, coeffs);

       int n_ranges = static_cast<int>(coeffs[0]);
       *cp_coeff_matrix = (double*) malloc(n_coeffs * n_ranges * sizeof(double));
       *temp_ranges     = (double*) malloc( (n_ranges+1) * sizeof(double));

       size_t index = 1;
       for (unsigned int j = 0; j < n_ranges; ++j) {
          (*temp_ranges)[j] = coeffs[index];
          if (j == n_ranges - 1) {
           (*temp_ranges)[j+1] = coeffs[index+1];
          }
          for (unsigned int k = 0; k < n_coeffs; ++k) {
             (*cp_coeff_matrix)[k+j*n_coeffs] = coeffs[index + 2 + k];
          }
          index += 11;
       }
       *n_coeffs_ret = n_coeffs;
       *n_ranges_ret = n_ranges;


      } else if (nasa7poly2) {
       int n_coeffs = 7;
       int n_ranges = 2;
       size_t species_index;
       int type;
       double tlow, thigh, pref;
       double* coeffs = static_cast<double*>(malloc((1+2*7)*sizeof(double))); // Assuming a maximum of 10 zones

       *cp_coeff_matrix = (double*) malloc(n_coeffs * n_ranges * sizeof(double));
       *temp_ranges     = (double*) malloc( (n_ranges+1) * sizeof(double));

       nasa7poly2->reportParameters(species_index, type, tlow, thigh, pref, coeffs);

       (*temp_ranges)[0] = tlow;
       (*temp_ranges)[1] = coeffs[0];
       (*temp_ranges)[2] = thigh;

       size_t index = 8; // ***
       for (unsigned int j = 0; j < n_ranges; ++j) {
          for (unsigned int k = 0; k < n_coeffs; ++k) {
             (*cp_coeff_matrix)[k+j*n_coeffs] = coeffs[index  + k];
          }
          index -= 7;  // *** these two lines are due to reportParam giving first high T coeffs and then low T coeffs
       }
       *n_coeffs_ret = n_coeffs;
       *n_ranges_ret = n_ranges;



      } else if (nasa7poly1) {
        std::cerr << "Cantera error: NASA7 with 1 temperature range not tested yet " << std::endl;
        int n_coeffs = 7;
        int n_ranges = 1;
        size_t species_index;
        int type;
        double tlow, thigh, pref;
        double* coeffs = static_cast<double*>(malloc((1+2*7)*sizeof(double))); // Assuming a maximum of 10 zones

        *cp_coeff_matrix = (double*) malloc(n_coeffs * n_ranges * sizeof(double));
        *temp_ranges     = (double*) malloc( (n_ranges+1) * sizeof(double));

        nasa7poly1->reportParameters(species_index, type, tlow, thigh, pref, coeffs);

        (*temp_ranges)[0] = tlow;
        (*temp_ranges)[1] = thigh;

        size_t index = 8; // ***
        for (unsigned int k = 0; k < n_coeffs; ++k) {
           (*cp_coeff_matrix)[k] = coeffs[k];
        }

        *n_coeffs_ret = n_coeffs;
        *n_ranges_ret = n_ranges;
     } else {
       std::cerr << "Cantera error: cp format not supported for specie " << i_specie << std::endl;
     }


}

void get_arrhenius_from_cantera_c(int *nsp_ret, int *nreacts_ret, double **A, double **b, double **Ea, double **thirdbody_eff, double **falloffcoeff) {
   auto sol     = newSolution("input_cantera.yaml");
   auto kin     = sol->kinetics();
   auto th      = sol->thermo();
   int  nsp     = th->nSpecies();
   int  nreacts = kin->nReactions();

   *A  = (double*) malloc(nreacts * 2 * sizeof(double));
   *b  = (double*) malloc(nreacts * 2 * sizeof(double));
   *Ea = (double*) malloc(nreacts * 2 * sizeof(double));
   *thirdbody_eff = (double*) malloc(nsp * nreacts * sizeof(double));
   *falloffcoeff = (double*) malloc(nreacts * 5 * sizeof(double));
   for (unsigned int i = 0; i < nreacts; i++) {
      auto reaction = kin->reaction(i);
      auto arrhenius = std::dynamic_pointer_cast<Cantera::ArrheniusRate>(reaction->rate());
      auto lindemann = std::dynamic_pointer_cast<Cantera::LindemannRate>(reaction->rate());
      auto troe = std::dynamic_pointer_cast<Cantera::TroeRate>(reaction->rate());
      auto sri = std::dynamic_pointer_cast<Cantera::SriRate>(reaction->rate());

      if (arrhenius) { //if reaction->rate() returns a pointer to an object that is of type Cantera::ArrheniusRate
         (*A)[i*2]  = arrhenius->preExponentialFactor();
         (*b)[i*2]  = arrhenius->temperatureExponent();
         (*Ea)[i*2] = arrhenius->activationEnergy();
         (*A)[i*2+1] = 0.0;
         (*b)[i*2+1]  = 0.0;
         (*Ea)[i*2+1] = 0.0;
         (*falloffcoeff)[i*5] = 0.0;
         (*falloffcoeff)[i*5+1] = 0.0;
         (*falloffcoeff)[i*5+2] = 0.0;
         (*falloffcoeff)[i*5+3] = 0.0;
         (*falloffcoeff)[i*5+4] = 0.0;
      } else if (lindemann) {
        (*A)[i*2] = lindemann->lowRate().preExponentialFactor();
        (*b)[i*2] = lindemann->lowRate().temperatureExponent();
        (*Ea)[i*2] = lindemann->lowRate().activationEnergy();
        (*A)[i*2+1] = lindemann->highRate().preExponentialFactor();
        (*b)[i*2+1] = lindemann->highRate().temperatureExponent();
        (*Ea)[i*2+1] = lindemann->highRate().activationEnergy();
        (*falloffcoeff)[i*5] = 0.0;
        (*falloffcoeff)[i*5+1] = 0.0;
        (*falloffcoeff)[i*5+2] = 0.0;
        (*falloffcoeff)[i*5+3] = 0.0;
        (*falloffcoeff)[i*5+4] = 0.0;
      } else if (troe) {
        (*A)[i*2] = troe->lowRate().preExponentialFactor();
        (*b)[i*2] = troe->lowRate().temperatureExponent();
        (*Ea)[i*2] = troe->lowRate().activationEnergy();
        (*A)[i*2+1] = troe->highRate().preExponentialFactor();
        (*b)[i*2+1] = troe->highRate().temperatureExponent();
        (*Ea)[i*2+1] = troe->highRate().activationEnergy();
        std::vector<double> c;
        troe->getFalloffCoeffs(c);
        if (c.size() == 3) {            //c size allowed: 3 or 4
                for (size_t j = 0; j < c.size(); ++j) {
                (*falloffcoeff)[i*5+j] = c[j];
                }
        (*falloffcoeff)[i*5+3] = -3.14;   //4th coeff (T2) is optional. If not provided, set a dummy value
        } else {
                for (size_t j = 0; j < c.size(); ++j) {
                (*falloffcoeff)[i*5+j] = c[j];
        }
        }
        (*falloffcoeff)[i*5+4] = 0.0;    //For Troe reactions only 4 coeffs are allowed. 5th is always zero
        } else if (sri) {
        (*A)[i*2] = sri->lowRate().preExponentialFactor();
        (*b)[i*2] = sri->lowRate().temperatureExponent();
        (*Ea)[i*2] = sri->lowRate().activationEnergy();
        (*A)[i*2+1] = sri->highRate().preExponentialFactor();
        (*b)[i*2+1] = sri->highRate().temperatureExponent();
        (*Ea)[i*2+1] = sri->highRate().activationEnergy();
        std::vector<double> c;
        sri->getFalloffCoeffs(c);
        if (c.size() == 3) {            //c size allowed: 3 or 5
                for (size_t j = 0; j < c.size(); ++j) {
                (*falloffcoeff)[i*5+j] = c[j];
                }
        (*falloffcoeff)[i*5+3] = 1.0;   //4th coeff (d) is optional. If not provided, set default value d = 1
        (*falloffcoeff)[i*5+4] = 0.0;   //5th coeff (e) is optional. If not provided, set default value e = 0
        } else if (c.size() == 5) {
                for (size_t j = 0; j < c.size(); ++j) {
                (*falloffcoeff)[i*5+j] = c[j];
        }
        }
      } else { //if reaction->rate() returns a pointer to an object that is not of type Cantera::ArrheniusRate
         (*A)[i*2]  = 0.0;
         (*b)[i*2]  = 0.0;
         (*Ea)[i*2] = 0.0;
         (*A)[i*2+1]  = 0.0;
         (*b)[i*2+1]  = 0.0;
         (*Ea)[i*2+1] = 0.0;
         (*falloffcoeff)[i*5] = 0.0;
         (*falloffcoeff)[i*5+1] = 0.0;
         (*falloffcoeff)[i*5+2] = 0.0;
         (*falloffcoeff)[i*5+3] = 0.0;
         (*falloffcoeff)[i*5+4] = 0.0;
      }

      auto third_body = std::dynamic_pointer_cast<Cantera::ThirdBody>(reaction->thirdBody());

      if (third_body){ // if ThirdBody has been instanciated for reaction(j) == if reaction(j) is a third body reaction
         for (int j = 0; j < nsp; j++) {
            const std::string name=th->speciesName(j);
//            if (third_body->efficiency(name) == 1) { // 1 is the default value but I want to switch to -1
//              (*thirdbody_eff)[i * nsp +j] = -1.0;
//            } else {
               (*thirdbody_eff)[i * nsp +j] = third_body->efficiency(name);
//            }
         }
      } else { // if ThirdBody has not been instanciated for reaction(j) == if reaction(j) is not a third body reaction
         for (int j = 0; j < nsp; j++) {
            (*thirdbody_eff)[i * nsp +j] = 1.0;
         }
      }
   }
  *nreacts_ret = nreacts;
  *nsp_ret = nsp;
  }
}
