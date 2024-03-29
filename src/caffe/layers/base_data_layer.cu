#include <vector>

#include "caffe/data_layers.hpp"

namespace caffe {

template <typename Dtype>
void BasePrefetchingDataLayer<Dtype>::Forward_gpu(
    const vector<Blob<Dtype>*>& bottom, vector<Blob<Dtype>*>* top) {
  // First, join the thread
  JoinPrefetchThread();
  // Copy the data
  int skip = 1;
  caffe_copy(prefetch_data_.count(), prefetch_data_.cpu_data(),
      (*top)[0]->mutable_gpu_data());
  if (this->output_labels_) {
    caffe_copy(prefetch_label_.count(), prefetch_label_.cpu_data(),
        (*top)[1]->mutable_gpu_data());
    skip = 2;
  }
  for (int i = 0; i < top->size() - skip; i++){
	  caffe_copy(prefetch_aux_data_[i]->count(), prefetch_aux_data_[i]->cpu_data(),
			  (*top)[i+skip]->mutable_gpu_data());
  }
  // Start a new prefetch thread
  CreatePrefetchThread();
}

INSTANTIATE_CLASS(BasePrefetchingDataLayer);

}  // namespace caffe
