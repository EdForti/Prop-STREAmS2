module cantera

    use iso_fortran_env
    use iso_c_binding
    integer, parameter :: rckind = REAL64

    interface importPhase
       module procedure ctfunc_importPhase
    endinterface importPhase

    interface nSpecies
       module procedure ctthermo_nSpecies
    endinterface nSpecies

    interface nReactions
       module procedure ctthermo_nReactions
    endinterface nReactions

    interface getMolecularWeights
       module procedure ctthermo_getMolecularWeights
    endinterface getMolecularWeights

    interface setState_TPY
       module procedure ctthermo_setState_TPY
       module procedure ctstring_setState_TPY
    endinterface setState_TPY

    interface viscosity
       module procedure ctrans_viscosity
    endinterface viscosity

    interface thermalConductivity
       module procedure ctrans_thermalConductivity
    endinterface thermalConductivity

    interface reactantstoichcoeff 
       module procedure ctkin_reactantStoichCoeff
    endinterface reactantstoichcoeff

    interface productstoichcoeff 
       module procedure ctkin_productStoichCoeff
    endinterface productstoichcoeff
     
    interface meanmolecularweight
       module procedure ctthermo_meanMolecularWeight 
    endinterface meanmolecularweight 

    interface cp_mass 
       module procedure ctthermo_cp_mass
    endinterface cp_mass 

    interface cv_mass 
       module procedure ctthermo_cv_mass
    endinterface cv_mass 

    interface cp_mole 
       module procedure ctthermo_cp_mole
    endinterface cp_mole 

    interface getreactiontype 
       module procedure ctkin_getReactionType 
    endinterface getreactiontype

    interface getspeciesname
       module procedure ctthermo_getSpeciesName
    endinterface getspeciesname

    interface getpartialmolarenthalpies
       module procedure ctthermo_getpartialmolarenthalpies
    endinterface getpartialmolarenthalpies

    interface getequilibriumconstants
       module procedure ctkin_getequilibriumconstants
    endinterface getequilibriumconstants
 
    interface getbindiffcoeffs
       module procedure ctrans_getBinDiffCoeffs
    endinterface getbindiffcoeffs

    type phase_t
    endtype

    contains

    type(phase_t) function ctfunc_importPhase(src, id, loglevel)
       implicit none
       character*(*), intent(in) :: src
       character*(*), intent(in), optional :: id
       integer, intent(in), optional :: loglevel

       print *, 'Warning calling importPhase_dummy: this must not happen!'
    end function ctfunc_importphase

    integer function ctthermo_nSpecies(self)
       implicit none
       type(phase_t), intent(inout) :: self

       print *, 'Warning calling nSpecies_dummy: this must not happen!'
    end function ctthermo_nSpecies

    integer function ctthermo_nReactions(self)
       implicit none
       type(phase_t), intent(inout) :: self

       print *, 'Warning calling nSpecies_dummy: this must not happen!'
    end function ctthermo_nReactions

    subroutine ctthermo_getMolecularWeights(self, mw)
       implicit none
       type(phase_t), intent(inout) :: self
       double precision, intent(out) :: mw(*)

       print *, 'Warning calling getMolecularWeights_dummy: this must not happen!'
    end subroutine ctthermo_getmolecularweights

   subroutine ctthermo_setState_TPY(self, t, p, y)
       implicit none
       type(phase_t), intent(inout) :: self
       double precision, intent(in) :: t
       double precision, intent(in) :: p
       double precision, intent(in) :: y(*)

       print *, 'Warning calling setState_TPY_dummy: this must not happen!'
    end subroutine ctthermo_setState_TPY

    subroutine ctstring_setState_TPY(self, t, p, y)
       implicit none
       type(phase_t), intent(inout) :: self
       double precision, intent(in) :: t
       double precision, intent(in) :: p
       character*(*), intent(in) :: y

       print *, 'Warning calling setState_TPY_dummy: this must not happen!'
    end subroutine ctstring_setState_TPY

    double precision function ctrans_viscosity(self)
       implicit none
       type(phase_t), intent(inout) :: self

       print *, 'Warning calling viscosity_dummy: this must not happen!'
    end function ctrans_viscosity

    double precision function ctrans_thermalConductivity(self)
       implicit none
       type(phase_t), intent(inout) :: self

       print *, 'Warning calling thermalConductivity_dummy: this must not happen!'
    end function ctrans_thermalConductivity

    double precision function ctkin_reactantStoichCoeff(self, k, i)
      implicit none
      type(phase_t), intent(in) :: self
      integer, intent(in) :: k
      integer, intent(in) :: i

       print *, 'Warning calling reactantStoichCoeff_dummy: this must not happen!'
    end function ctkin_reactantstoichcoeff

    double precision function ctkin_productStoichCoeff(self, k, i)
      implicit none
      type(phase_t), intent(in) :: self
      integer, intent(in) :: k
      integer, intent(in) :: i

       print *, 'Warning calling productstoichcoeff_dummy: this must not happen!'
    end function ctkin_productstoichcoeff

    double precision function ctthermo_meanMolecularWeight(self)
      implicit none
      type(phase_t), intent(in) :: self

       print *, 'Warning calling meanMolecularWeight_dummy: this must not happen!'
    end function ctthermo_meanMolecularWeight 

    double precision function ctthermo_cp_mass(self)
      implicit none
      type(phase_t), intent(in) :: self

       print *, 'Warning calling cp_mass_dummy: this must not happen!'
    end function ctthermo_cp_mass

    double precision function ctthermo_cv_mass(self)
      implicit none
      type(phase_t), intent(in) :: self

       print *, 'Warning calling cv_mass_dummy: this must not happen!'
    end function ctthermo_cv_mass


    double precision function ctthermo_cp_mole(self)
      implicit none
      type(phase_t), intent(in) :: self

       print *, 'Warning calling cp_mole_dummy: this must not happen!'
    end function ctthermo_cp_mole

    subroutine ctkin_getReactionType(self, i, buf)
      implicit none
      type(phase_t), intent(inout) :: self
      integer, intent(in) :: i
      character*(*), intent(out) :: buf

       print *, 'Warning calling getReactionType_dummy: this must not happen!'
    end subroutine ctkin_getReactionType

    subroutine ctthermo_getSpeciesName(self, k, nm)
      implicit none
      type(phase_t), intent(inout) :: self
      integer, intent(in) :: k
      character*(*), intent(out) :: nm

       print *, 'Warning calling getSpeciesName_dummy: this must not happen!'
    end subroutine ctthermo_getSpeciesName

    subroutine ctthermo_getPartialMolarEnthalpies(self, hbar)
      implicit none
      type(phase_t), intent(inout) :: self
      double precision, intent(out) :: hbar(*)
  
       print *, 'Warning calling getPartialMolarEnthalpies_dummy: this must not happen!'
    end subroutine ctthermo_getPartialMolarEnthalpies

    subroutine ctkin_getEquilibriumConstants(self, kc)
      implicit none
      type(phase_t), intent(inout) :: self
      double precision, intent(out) :: kc(*)
   
        print *, 'Warning calling getEquilibriumConstants_dummy: this must not happen!'
    end subroutine ctkin_getequilibriumconstants

    subroutine ctrans_getBinDiffCoeffs(self, ld, d)
      implicit none
      type(phase_t), intent(inout) :: self
      integer, intent(in) :: ld
      double precision, intent(out) :: d(*)
      
       print *, 'Warning calling getBinDiffCoeffs_dummy: this must not happen!'
    end subroutine ctrans_getBinDiffCoeffs
 
endmodule cantera
