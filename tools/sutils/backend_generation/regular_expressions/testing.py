import unittest
from regex import FortranRegularExpressions,CudaFRegularExpressions

FRegex = FortranRegularExpressions()
CFRegex = CudaFRegularExpressions()

class TestFortranRegex(unittest.TestCase):
  maxDiff = None
  def test_ALLOC_FOR_VAR_RE(self):
    tests = ["allocate(self%winf_gpu(nv))","allocate(self%winf_past_shock_gpu(nv))","allocate(self%w_aux_gpu(1-ng:nx+ng, 1-ng:ny+ng, 1-ng:nz+ng, nv_aux))",
             "allocate(self%wmean_gpu(1-self%grid%ng:self%field%nx+self%grid%ng+1, 1:self%field%ny,4))",
             "allocate(self%wx_spec_gpu(self%grid%nxmax,self%grid%nymax/self%field%nblocks(1),nz,self%equation_base%nv_spec))"]
    group2 = ["self%winf_gpu(nv)","self%winf_past_shock_gpu(nv)","self%w_aux_gpu(1-ng:nx+ng, 1-ng:ny+ng, 1-ng:nz+ng, nv_aux)",
              "self%wmean_gpu(1-self%grid%ng:self%field%nx+self%grid%ng+1, 1:self%field%ny,4)",
              "self%wx_spec_gpu(self%grid%nxmax,self%grid%nymax/self%field%nblocks(1),nz,self%equation_base%nv_spec)"]
    group1 = tests
    for idx,test in enumerate(tests):
      match = FRegex.ALLOC_FOR_VAR_RE.match(test)
      self.assertIsNotNone(match)
      self.assertEqual(match.group(1), group1[idx])
      self.assertEqual(match.group(2), group2[idx])
  def test_2(self):
    # ALLOC_VAR_ATTRIB_SINGLE_RE
    # Remove whitespace (have not testes with whitespace!)
    # ALLOCATE with only one array
    tests = ["self%winf_gpu(nv)","self%winf_past_shock_gpu(nv)","self%w_aux_gpu(1-ng:nx+ng, 1-ng:ny+ng, 1-ng:nz+ng, nv_aux)",
              "self%wmean_gpu(1-self%grid%ng:self%field%nx+self%grid%ng+1, 1:self%field%ny,4)",
              "self%wx_spec_gpu(self%grid%nxmax,self%grid%nymax/self%field%nblocks(1),nz,self%equation_base%nv_spec)"]
    group1 = ["winf_gpu","winf_past_shock_gpu","w_aux_gpu",
              "wmean_gpu",
              "wx_spec_gpu"]
    group2 = ["nv","nv","1-ng:nx+ng, 1-ng:ny+ng, 1-ng:nz+ng, nv_aux",
              "1-self%grid%ng:self%field%nx+self%grid%ng+1, 1:self%field%ny,4",
              "self%grid%nxmax,self%grid%nymax/self%field%nblocks(1),nz,self%equation_base%nv_spec"]
    for idx,test in enumerate(tests):
      match = FRegex.ALLOC_VAR_ATTRIB_SINGLE_RE.findall(test)
      for m in match:
        self.assertIsNotNone(m)
        self.assertEqual(m[0], group1[idx])
        self.assertEqual(m[1], group2[idx])
  def test_3(self):
    # ALLOC_VAR_ATTRIB_MULT_RE
    # Remove whitespace (have not tested with whitespace!)
    # ALLOCATE with only multiple arrays
    tests = ["self%x_gpu(1-ng:nx+ng),self%y_gpu(1-ng:ny+ng),self%z_gpu(1-ng:nz+ng)",
              "self%dcsidx_gpu(nx),self%dcsidx2_gpu(nx),self%dcsidxs_gpu(nx)"]
    group1 = [("x_gpu","y_gpu","z_gpu"),("dcsidx_gpu","dcsidx2_gpu","dcsidxs_gpu")]
    group2 = [("1-ng:nx+ng","1-ng:ny+ng","1-ng:nz+ng"),("nx","nx","nx")]
    
    for idx,test in enumerate(tests):
      match = FRegex.ALLOC_VAR_ATTRIB_MULT_RE.findall(test)
      for i,m in enumerate(match):
        self.assertIsNotNone(m)
        self.assertEqual(m[0], group1[idx][i])
        self.assertEqual(m[1], group2[idx][i])
  def test_4(self):
    # DIM_FORMAT_RE
    tests = ["1-self%grid%ng:self%field%nx+self%grid%ng+1,1:self%field%ny,4",
             "6,self%equation_base%num_probe",
             "2,2,2,self%equation_base%num_probe",
             "self%grid%nxmax,self%grid%nymax/self%field%nblocks(1),nz,self%equation_base%nv_spec",
             "nx,self%grid%nymax/self%field%nblocks(3),self%grid%nzmax,self%equation_base%nv_spec"
             ]
    substitute = ["1-ng:nx+ng+1,1:ny,4",
                  "6,num_probe",
                  "2,2,2,num_probe",
                  "nxmax,nymax/nblocks(1),nz,nv_spec",
                  "nx,nymax/nblocks(3),nzmax,nv_spec"]
    
    for idx,test in enumerate(tests):
      match = FRegex.DIM_FORMAT_RE.sub("",test)
      self.assertIsNotNone(match)
      self.assertEqual(match, substitute[idx])
  def test_5(self):
    # ARRAY_TYPE_DIM_RE
    test_string = "real(rkind), allocatable, dimension(:,:,:,:), device :: w_gpu\n \
             real(rkind), allocatable, dimension(:,:,:,:)         :: w_t\n \
             real(rkind), dimension(:,:,:,:), device, allocatable :: wbuf1s_gpu, wbuf2s_gpu, wbuf3s_gpu, &\n \
                                                                wbuf4s_gpu, wbuf5s_gpu, wbuf6s_gpu\n \
             real(rkind), dimension(:,:,:,:), device, allocatable :: wbuf1r_gpu, wbuf2r_gpu, wbuf3r_gpu, &\n \
                                                                wbuf4r_gpu, wbuf5r_gpu, wbuf6r_gpu\n \
             real(rkind), dimension(:,:,:,:), allocatable, device :: wbuf1s_c_gpu, wbuf2s_c_gpu, wbuf3s_c_gpu, wbuf4s_c_gpu\n \
             real(rkind), dimension(:,:,:,:), allocatable, device :: wbuf1r_c_gpu, wbuf2r_c_gpu, wbuf3r_c_gpu, wbuf4r_c_gpu\n \
             real(rkind), allocatable, dimension(:), device :: dcsidx_gpu, dcsidx2_gpu, dcsidxs_gpu\n \
             real(rkind), allocatable, dimension(:), device :: detady_gpu, detady2_gpu, detadys_gpu\n \
             real(rkind), allocatable, dimension(:), device :: dzitdz_gpu, dzitdz2_gpu, dzitdzs_gpu\n \
             real(rkind), allocatable, dimension(:), device :: dzitdzc_gpu, dzitdzn_gpu\n \
             real(rkind), allocatable, dimension(:), device :: detadyc_gpu, detadyn_gpu\n \
             real(rkind), allocatable, dimension(:), device :: detady_2nd_gpu,dzitdz_2nd_gpu\n \
             real(rkind), allocatable, dimension(:), device :: x_gpu, y_gpu, z_gpu, yn_gpu\n \
             real(rkind), allocatable, dimension(:,:), device :: coeffs_wallinterp_gpu\n \
             real(rkind), allocatable, dimension(:,:,:,:), device :: gplus_x_gpu, gminus_x_gpu\n \
             real(rkind), allocatable, dimension(:,:,:,:), device :: gplus_y_gpu, gminus_y_gpu\n \
             real(rkind), allocatable, dimension(:,:,:,:), device :: gplus_z_gpu, gminus_z_gpu\n \
             integer, allocatable, dimension(:,:,:), device :: fluid_mask_gpu\n \
             integer, allocatable, dimension(:,:,:), device :: fluid_mask_trans_gpu"
    arrays = ["w_gpu","w_t","dcsidx_gpu","z_gpu","yn_gpu","wbuf2r_c_gpu","wbuf5s_gpu","fluid_mask_gpu","gminus_y_gpu"]
    group1 = ["real","real","real","real","real","real","real","integer","real"]
    group2 = ["dimension(:,:,:,:)","dimension(:,:,:,:)","dimension(:)","dimension(:)","dimension(:)","dimension(:,:,:,:)","dimension(:,:,:,:)",
              "dimension(:,:,:)","dimension(:,:,:,:)"]
    
    for idx,test in enumerate(arrays):
      match = FRegex.ARRAY_TYPE_DIM_RE(test).search(test_string)
      
      self.assertIsNotNone(match)
      self.assertEqual(group1[idx], match.group(1))
      self.assertEqual(group2[idx], match.group(2))
      
  def test_6(self):
    # CAPTURE_COMMENT_RE
    test_string = "    use streams_field_object, only : field_object\n"\
	"    use streams_parameters\n"\
	"    use MPI\n"\
	"    use CUDAFOR\n\n"\
	"    implicit none\n"\
	"    private\n"\
	"    public :: base_gpu_object\n\n"\
	"    type :: base_gpu_object\n"\
	"        type(field_object), pointer :: field=>null() \n"\
	"        ! Replica of field and grid sizes\n"\
	"        integer :: nx, ny, nz, ng, nv\n"\
	"        ! MPI data\n"\
	"        integer(ikind)              :: myrank=0_ikind       !< MPI rank process.\n"\
	"        integer(ikind)              :: nprocs=1_ikind       !< Number of MPI processes.\n"\
	"        logical                     :: masterproc  \n"\
	"        integer(ikind)              :: mpi_err=0_ikind      !< Error traping flag.\n"\
	"        integer(ikind)              :: mydev=0_ikind        !< My GPU rank.\n"\
	"        integer(ikind)              :: local_comm=0_ikind   !< Local communicator.\n"\
	"        ! GPU data\n"\
	"        real(rkind), allocatable, dimension(:,:,:,:), device :: w_gpu\n"\
	"        real(rkind), allocatable, dimension(:,:,:,:)         :: w_t\n"
    result = "    use streams_field_object, only : field_object\n"\
	"    use streams_parameters\n"\
	"    use MPI\n"\
	"    use CUDAFOR\n\n"\
	"    implicit none\n"\
	"    private\n"\
	"    public :: base_gpu_object\n\n"\
	"    type :: base_gpu_object\n"\
	"        type(field_object), pointer :: field=>null() \n"\
	"        \n"\
	"        integer :: nx, ny, nz, ng, nv\n"\
	"        \n"\
	"        integer(ikind)              :: myrank=0_ikind       \n"\
	"        integer(ikind)              :: nprocs=1_ikind       \n"\
	"        logical                     :: masterproc  \n"\
	"        integer(ikind)              :: mpi_err=0_ikind      \n"\
	"        integer(ikind)              :: mydev=0_ikind        \n"\
	"        integer(ikind)              :: local_comm=0_ikind   \n"\
	"        \n"\
	"        real(rkind), allocatable, dimension(:,:,:,:), device :: w_gpu\n"\
	"        real(rkind), allocatable, dimension(:,:,:,:)         :: w_t\n" 

    
    match = FRegex.CAPTURE_COMMENT_RE.sub("",test_string)
      
    self.assertIsNotNone(match)
    self.assertEqual(match, result)
  def test_7(self):
    # ALL_SUBROUTINES_RE
    test_str = "module streams_kernels_gpu\n\n"\
	"    use streams_parameters, only : rkind, ikind, REAL64\n"\
	"    use CUDAFOR\n"\
	"    implicit none\n\n"\
	"contains\n\n"\
	"    subroutine init_flux_cuf(nx, ny, nz, nv, fl_gpu, fln_gpu, rhodt) \n"\
	"        integer :: nx, ny, nz, nv\n"\
	"        real(rkind) :: rhodt\n"\
	"        real(rkind), dimension(1:,1:,1:,1:), intent(inout), device :: fl_gpu, fln_gpu\n"\
	"        integer :: i,j,k,m,iercuda\n"\
	"        !$cuf kernel do(3) <<<*,*>>> \n"\
	"         do k=1,nz\n"\
	"          do j=1,ny\n"\
	"           do i=1,nx\n"\
	"            do m=1,nv\n"\
	"             fln_gpu(i,j,k,m) = - rhodt * fl_gpu(i,j,k,m)\n"\
	"             fl_gpu(i,j,k,m)  = 0._rkind\n"\
	"            enddo\n"\
	"           enddo\n"\
	"          enddo\n"\
	"         enddo\n"\
	"        !@cuf iercuda=cudaDeviceSynchronize()\n"\
	"!        \n"\
	"    endsubroutine init_flux_cuf\n"
    group1 = "    subroutine init_flux_cuf(nx, ny, nz, nv, fl_gpu, fln_gpu, rhodt) \n"\
	"        integer :: nx, ny, nz, nv\n"\
	"        real(rkind) :: rhodt\n"\
	"        real(rkind), dimension(1:,1:,1:,1:), intent(inout), device :: fl_gpu, fln_gpu\n"\
	"        integer :: i,j,k,m,iercuda\n"\
	"        !$cuf kernel do(3) <<<*,*>>> \n"\
	"         do k=1,nz\n"\
	"          do j=1,ny\n"\
	"           do i=1,nx\n"\
	"            do m=1,nv\n"\
	"             fln_gpu(i,j,k,m) = - rhodt * fl_gpu(i,j,k,m)\n"\
	"             fl_gpu(i,j,k,m)  = 0._rkind\n"\
	"            enddo\n"\
	"           enddo\n"\
	"          enddo\n"\
	"         enddo\n"\
	"        !@cuf iercuda=cudaDeviceSynchronize()\n"\
	"!        \n"\
	"    endsubroutine init_flux_cuf\n"
    group2 = "init_flux_cuf"
    match = FRegex.ALL_SUBROUTINES_RE.search(test_str)
    group1_match = match.group(1)
    group2_match = match.group(2)
    self.assertIsNotNone(match)
    self.assertEqual(group1_match, group1)
    self.assertEqual(group2_match, group2)
  
class TestCudaFortranRegex(unittest.TestCase):
  maxDiff = None
  def test_1(self):
    # ARRAY_TYPE_DIM_RE
    test_string = "    subroutine update_flux_cuf(nx, ny, nz, nv, fl_gpu, fln_gpu, gamdt) \n"\
	"        integer :: nx, ny, nz, nv\n"\
	"        real(rkind) :: gamdt\n"\
	"        real(rkind), dimension(1:,1:,1:,1:), intent(in), device :: fl_gpu\n"\
	"        real(rkind), dimension(1:,1:,1:,1:), intent(inout), device :: fln_gpu ! GPU arrays\n"\
	"        integer :: i,j,k,m,iercuda\n"\
	"        !$cuf kernel do(3) <<<*,*>>> !No comment\n"\
	"         do k=1,nz\n"\
	"          do j=1,ny\n"\
	"           do i=1,nx\n"\
	"            do m=1,nv\n"\
	"             fln_gpu(i,j,k,m) = fln_gpu(i,j,k,m)-gamdt*fl_gpu(i,j,k,m)\n"\
	"            enddo\n"\
	"           enddo\n"\
	"          enddo\n"\
	"         enddo\n"\
	"        !@cuf iercuda=cudaDeviceSynchronize()\n"\
	"! End of the kernel\n"\
	"    endsubroutine update_flux_cuf"
    result = "    subroutine update_flux_cuf(nx, ny, nz, nv, fl_gpu, fln_gpu, gamdt) \n"\
	"        integer :: nx, ny, nz, nv\n"\
	"        real(rkind) :: gamdt\n"\
	"        real(rkind), dimension(1:,1:,1:,1:), intent(in), device :: fl_gpu\n"\
	"        real(rkind), dimension(1:,1:,1:,1:), intent(inout), device :: fln_gpu \n"\
	"        integer :: i,j,k,m,iercuda\n"\
	"        !$cuf kernel do(3) <<<*,*>>> \n"\
	"         do k=1,nz\n"\
	"          do j=1,ny\n"\
	"           do i=1,nx\n"\
	"            do m=1,nv\n"\
	"             fln_gpu(i,j,k,m) = fln_gpu(i,j,k,m)-gamdt*fl_gpu(i,j,k,m)\n"\
	"            enddo\n"\
	"           enddo\n"\
	"          enddo\n"\
	"         enddo\n"\
	"        !@cuf iercuda=cudaDeviceSynchronize()\n"\
	"\n"\
	"    endsubroutine update_flux_cuf" 

    
    match = CFRegex.CAPTURE_COMMENT_RE.sub("",test_string)
      
    self.assertIsNotNone(match)
    self.assertEqual(match, result)

  def test_2(self):
    # GLOBAL_KERNEL_BOUND_RE
    # Should only matches the first occurence of a global kernel
    # Using it to get serial code by removing the thread id like below:
    """
    attributes (global) subroutine......
     ----declarations----
     
     i = blockDim.....
     j = blockDim.....
     
     if(i>__ .or. j>__) return
     
     ----------serial code-------------
    
     end if
    end subroutine
    
    Should remove:
    
     if(i>__ .or. j>__) return
    
    """
    test_string_array = "if (j > ny .or. i > eul_imax) return"
    
    match_j = CFRegex.GLOBAL_KERNEL_BOUND_RE("j").search(test_string_array)
    match_i = CFRegex.GLOBAL_KERNEL_BOUND_RE("i").search(test_string_array)

    self.assertIsNotNone(match_j)
    self.assertIsNotNone(match_i)
    self.assertEqual(match_j.group(1),"ny")
    self.assertEqual(match_i.group(1),"eul_imax")
    
if __name__ == '__main__':
  unittest.main(exit=False)