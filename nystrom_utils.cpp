#include "nystrom_utils.hpp"

using namespace El;

template<typename T>
std::ostream &operator<<(std::ostream &stream, std::vector<T> ob)
{
	for(int i=0;i<ob.size();i++){
	  stream << ob[i] << '\n';
	}
	return stream;
}

std::vector<int> loc_exscan(std::vector<int> x){
	int ll = x.size();
	std::vector<int> y(ll);
	y[0] = 0;

	for(int i=1;i<ll;i++){
		y[i] = y[i-1] + x[i-1];
	}
	
	return y;
}

void make_vec_from_loc(std::vector<double>& loc_vec, El::DistMatrix<double,VC,STAR> el_vec){
	// Initialize
	int nlocal = loc_vec.size();
	int r,q,rq; //Grid sizes
	int nbigs; //Number of large sends (i.e. send 1 extra data point)
	int pstart; // p_id of nstart
	int p = El::mpi::WorldRank(); //p_id
	int send_size; // base send size
	bool print = p == -1; 


	// Get Grid and associated params
	const El::Grid* g = &(el_vec.Grid());
	r = g->Height();
	q = g->Width();
	rq = r * q;
	MPI_Comm comm = (g->Comm()).comm;

	// Find others local sizes
	int rank = mpi::WorldRank();
	std::vector<int> sends(rq,nlocal);
	std::vector<int> recvs(rq);
	std::vector<int> offsets(rq);
	mpi::AllToAll(&sends[0],1,&recvs[0],1,mpi::COMM_WORLD);
	
	
	// Get global sizes and offsets
	//std::partial_sum(sends.begin(),sends.end(),offsets.begin());
	//int nstart;
	//if(rank==0){
	//	nstart = 0;
	//}else{
	//	nstart = offsets[rank - 1];
	//}
	//int gsize = offsets[comms-1];	
	offsets = loc_exscan(recvs);
	int nstart = offsets[rank];
	int gsize = offsets[rq-1] + recvs[rq-1];

	// Resize el_vec
	el_vec.Resize(gsize,1);
	
	//Find processor that nstart belongs to, number of larger sends
	pstart = nstart % rq; //int div
	nbigs = nlocal % rq;
	send_size = nlocal/rq;
	
	if(print){
		std::cout << "r: " << r << " q: " << q <<std::endl;
		std::cout << "nstart: " << nstart << std::endl;
		std::cout << "ps: " << pstart << std::endl;
		std::cout << "nbigs: " << nbigs << std::endl;
		std::cout << "send_size: " << send_size << std::endl;
	}

	// Make send_lengths
	std::vector<int> send_lengths(rq);
	std::fill(send_lengths.begin(),send_lengths.end(),send_size);
	if(nbigs >0){
		for(int j=0;j<nbigs;j++){
			send_lengths[(pstart + j) % rq] += 1;
		}
	}

	// Make send_disps
	std::vector<int> send_disps = loc_exscan(send_lengths);

	// Make send_data
	std::vector<double> send_data(nlocal);
	for(int proc=0;proc<rq;proc++){
		int offset = send_disps[proc];
		int base_idx = (proc - pstart + rq) % rq; 
		for(int j=0; j<send_lengths[proc]; j++){
			int idx = base_idx + (j * rq);
			send_data[offset + j] = loc_vec[idx];
		}
	}

	// Do all2all to get recv_lengths
	std::vector<int> recv_lengths(rq);
	MPI_Alltoall(&send_lengths[0], 1, MPI_INT, &recv_lengths[0], 1, MPI_INT,comm);

	// Scan to get recv_disps
	std::vector<int> recv_disps = loc_exscan(recv_lengths);

	// Do all2allv to get data on correct processor
	double * recv_data = el_vec.Buffer();
	MPI_Alltoallv(&send_data[0],&send_lengths[0],&send_disps[0],MPI_DOUBLE, \
			&recv_data[0],&recv_lengths[0],&recv_disps[0],MPI_DOUBLE,comm);

	if(print){
		std::cout << "Send data: " <<std::endl << send_data <<std::endl;
		std::cout << "Send lengths: " <<std::endl << send_lengths <<std::endl;
		std::cout << "Send disps: " <<std::endl << send_disps <<std::endl;
		std::cout << "Recv data: " <<std::endl << recv_data <<std::endl;
		std::cout << "Recv lengths: " <<std::endl << recv_lengths <<std::endl;
		std::cout << "Recv disps: " <<std::endl << recv_disps <<std::endl;
	}
	

}

