! Compile with:
!
! $ mpif90 -fcheck=all pmatmul.f90
! $ mpiexec -n 1 ./a.out

module parallel_matmul
use mpi
implicit none
integer, parameter :: dp = kind(0.d0)
contains
subroutine pmatmul(A, x, y)
    real(dp), intent(inout) :: A(:, :), x(:), y(:)
    integer :: i, n, myrank, numprocs, rowstart, rowend, rows_per_rank, ierr

    call MPI_COMM_RANK(MPI_COMM_WORLD, myrank, ierr)
    call MPI_COMM_SIZE(MPI_COMM_WORLD, numprocs, ierr)

    n = size(x)

    if (myrank == 0) then
        print *, "nproc =", numprocs
        if (mod(n, numprocs) /= 0) then
            print *, 'Error: number of rows is is not divisible by the number of processes'
            write (*,*) 'n = ', n, 'but there are', numprocs, 'processes'
            call MPI_ABORT(MPI_COMM_WORLD, 1, ierr)
        end if
    end if

    rows_per_rank = n / numprocs
    rowstart = 1 + rows_per_rank * myrank
    rowend = min(n, rowstart + rows_per_rank - 1)

    ! Scatter matrix A and vector x among processes
    call MPI_SCATTER(A, rows_per_rank * n, &
     MPI_DOUBLE_PRECISION, A(:,rowstart), rows_per_rank * n, &
     MPI_DOUBLE_PRECISION, 0, MPI_COMM_WORLD, ierr)
    call MPI_BCAST(x, n, MPI_DOUBLE_PRECISION, 0, MPI_COMM_WORLD, ierr)
    call MPI_BCAST(y, n, MPI_DOUBLE_PRECISION, 0, MPI_COMM_WORLD, ierr)

    ! Compute partial dot products
    do i = rowstart, rowend
        y(i) = dot_product(A(:,i), x(:))
    end do

    ! Gather result vectors from all processes
    call MPI_GATHER(y(rowstart:rowend), (rowend-rowstart+1), &
     MPI_DOUBLE_PRECISION, y, (rowend-rowstart+1), MPI_DOUBLE_PRECISION, &
     0, MPI_COMM_WORLD, ierr)

end subroutine
end module
