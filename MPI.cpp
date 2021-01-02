#include <stdio.h>
#include <stdlib.h>
#include "mpi.h"
#include <string.h>
#ifndef W
#define W 20                                    // Width
#endif
int main(int argc, char **argv) {
	int L = atoi(argv[1]);                        // Length
	int iteration = atoi(argv[2]);                // Iteration
	srand(atoi(argv[3]));                         // Seed

	MPI_Init(&argc, &argv);

	// Get the number of processes
	int world_size;
	MPI_Comm_size(MPI_COMM_WORLD, &world_size);

	// Get the rank of the process
	int world_rank;
	MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

	float d = (float)random() / RAND_MAX * 0.2;  // Diffusivity
	int *temp = malloc(L*W * sizeof(int));          // Current temperature
	int *next = malloc(L*W * sizeof(int));          // Next time step
	int minArr[world_size];
	MPI_Status  status;
	MPI_Request request;

	int begin = (L / world_size) * world_rank;
	int end = (L / world_size) * (world_rank + 1);

	// Init temp
	for (int i = 0; i < L; i++) {
		for (int j = 0; j < W; j++) {
			temp[i*W + j] = random() >> 3;
		}
	}

	int count = 0, balance = 0;
	while (iteration--) {     // Compute with up, left, right, down points
		balance = 1;
		count++;

		// =============================================================================================
		// Data Transfer

		if (world_rank == 0) {
			MPI_Isend(temp + (end - 1) * W, W, MPI_INT, (world_rank + 1), count, MPI_COMM_WORLD, &request);
			MPI_Recv(temp + (end * W), W, MPI_INT, (world_rank + 1), count, MPI_COMM_WORLD, &status);
		}

		else if (world_rank > 0 && world_rank < (world_size - 1)) {
			MPI_Isend(temp + (end - 1) * W, W, MPI_INT, (world_rank + 1), count, MPI_COMM_WORLD, &request);
			MPI_Isend(temp + (begin * W), W, MPI_INT, (world_rank - 1), count, MPI_COMM_WORLD, &request);
			MPI_Recv(temp + end * W, W, MPI_INT, (world_rank + 1), count, MPI_COMM_WORLD, &status);
			MPI_Recv(temp + (begin - 1) * W, W, MPI_INT, (world_rank - 1), count, MPI_COMM_WORLD, &status);
		}

		else if (world_rank == (world_size - 1)) {
			MPI_Isend(temp + begin * W, W, MPI_INT, (world_rank - 1), count, MPI_COMM_WORLD, &request);
			MPI_Recv(temp + (begin - 1) * W, W, MPI_INT, (world_rank - 1), count, MPI_COMM_WORLD, &status);
		}

		// =============================================================================================

		for (int i = begin; i < end; i++) {
			for (int j = 0; j < W; j++) {
				float t = temp[i * W + j] / d;
				t += temp[i * W + j] * -4;
				t += temp[(i - 1 < 0 ? 0 : i - 1) * W + j];
				t += temp[(i + 1 >= L ? i : i + 1) * W + j];
				t += temp[i*W + (j - 1 < 0 ? 0 : j - 1)];
				t += temp[i*W + (j + 1 >= W ? j : j + 1)];
				t *= d;
				next[i * W + j] = t;
				if (next[i * W + j] != temp[i * W + j]) {
					balance = 0;
				}
			}
		}

		MPI_Barrier(MPI_COMM_WORLD);
		MPI_Bcast(&balance, 1, MPI_INT, 0, MPI_COMM_WORLD);
		if (balance) {
			break;
		}

		int *tmp = temp;
		temp = next;
		next = tmp;
	}

	int min = temp[begin * W];
	for (int i = begin; i < end; i++) {
		for (int j = 0; j < W; j++) {
			if (temp[i*W + j] < min) {
				min = temp[i*W + j];
			}
		}
	}

	MPI_Gather(&min, 1, MPI_INT, minArr, 1, MPI_INT, 0, MPI_COMM_WORLD);

	if (world_rank == 0) {
		int worldMin = minArr[0];
		for (int i = 0; i < world_size; i++) {
			if (minArr[i] < worldMin) {
				worldMin = minArr[i];
			}
		}
		printf("Size: %d*%d, Iteration: %d, Min Temp: %d\n", L, W, count, worldMin);
	}

	MPI_Finalize();
	return 0;
}