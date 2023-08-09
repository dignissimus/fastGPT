program test_parallel_matmul
use mpi
use parallel_matmul

implicit none

integer :: n ! matrix size
integer :: i, j, ierr, myrank
! integer, parameter :: dp = kind(0.d0)
real(kind(0.d0)), allocatable :: A(:,:), x(:), y(:), z(:)
n = 1000

! Initialize MPI environment
call MPI_INIT(ierr)
call MPI_COMM_RANK(MPI_COMM_WORLD, myrank, ierr)

allocate(A(n,n), x(n), y(n), z(n))

! Initialize matrix A and vector x
do i = 1, n
    x(i) = 1
    do j = 1, n
        A(i,j) = 1.0d0/(i+j-1.0d0)
    end do
end do


! Calculate the matrix-vector multiplication in parallel
call pmatmul(A, x, y)

! Finalize MPI environment
call MPI_FINALIZE(ierr)

! Calculate the matrix-vector multiplication using matmul
z = matmul(A, x)

if (myrank == 0 ) then
    ! Print the result vector y from rank 0
    write(*,*) 'Result vector y(:10):'
    write(*,*) y(:10)
    write(*,*) 'Actual vector z(:10):'
    write(*,*) z(:10)
    write(*,*) 'Total absolute error: '
    write(*,*) sum(abs(y - z))
    if (sum(abs(y - z)) > 1e-5) then
        error stop
    end if
end if

end program
