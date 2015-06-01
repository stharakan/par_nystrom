#include "askit_el_utils.hpp"


void setDistMatrix(askit::fksData fdata, El::DistMatrix<double> el_data){
	int loc_width = fdata.numof_points;
	int loc_height = fdata.dim;

	for(int j=0;j<loc_width;j++){
		int base_idx = j * loc_height;

		for(int i=0;i<loc_height;i++){
			int data_idx = base_idx + i;
			int glob_idx = (int) fdata.gids[j];
			
			double val = fdata.X[data_idx];

			el_data.Set(glob_idx,j,val);
		}
	}


}
