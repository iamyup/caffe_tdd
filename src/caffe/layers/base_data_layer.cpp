#include <string>
#include <vector>

#include "caffe/data_layers.hpp"
#include "caffe/util/io.hpp"

namespace caffe {

template <typename Dtype>
BaseDataLayer<Dtype>::BaseDataLayer(const LayerParameter& param)
    : Layer<Dtype>(param),
      transform_param_(param.transform_param()),
      data_transformer_(transform_param_) {
}

template <typename Dtype>
void BaseDataLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top) {
  if (top->size() == 1) {
    output_labels_ = false;
  } else {
    output_labels_ = true;
  }
  DataLayerSetUp(bottom, top);
  // The subclasses should setup the datum channels, height and width
  CHECK_GT(datum_channels_, 0);
  CHECK_GT(datum_height_, 0);
  CHECK_GT(datum_width_, 0);
  if (transform_param_.crop_size() > 0) {
    CHECK_GE(datum_height_, transform_param_.crop_size());
    CHECK_GE(datum_width_, transform_param_.crop_size());
  }
  // check if we want to have mean
  if (transform_param_.has_mean_file()) {
    const string& mean_file = transform_param_.mean_file();
    LOG(INFO) << "Loading mean file from" << mean_file;
    BlobProto blob_proto;
    ReadProtoFromBinaryFileOrDie(mean_file.c_str(), &blob_proto);
    data_mean_.FromProto(blob_proto);
    LOG(INFO)<< data_mean_.num() << " " << data_mean_.channels() << "  "
    << data_mean_.height()<< " " << data_mean_.width();
    CHECK_GE(data_mean_.num(), 1);
    CHECK_GE(data_mean_.channels(), datum_channels_);
    CHECK_GE(data_mean_.height(),transform_param_.crop_size());
    CHECK_GE(data_mean_.width(),transform_param_.crop_size());
    // CHECK_GE(data_mean_.height(), datum_height_);
    // CHECK_GE(data_mean_.width(), datum_width_);
  } else {
    // Simply initialize an all-empty mean.
    data_mean_.Reshape(1, datum_channels_, transform_param_.crop_size(), transform_param_.crop_size());
    Dtype* mean_prt = data_mean_.mutable_cpu_data();
    LOG(INFO) << transform_param_.is_flow();
    if (transform_param_.is_flow()==false){
      for (int i = 0; i<  transform_param_.crop_size()*transform_param_.crop_size(); i++)
        mean_prt[i] = 104;
      for (int i = 0; i<  transform_param_.crop_size()*transform_param_.crop_size(); i++)
        mean_prt[i+transform_param_.crop_size()*transform_param_.crop_size()] = 117;
      for (int i = 0; i<  transform_param_.crop_size()*transform_param_.crop_size(); i++)
        mean_prt[i+2*transform_param_.crop_size()*transform_param_.crop_size()] = 123;

      LOG(INFO) << mean_prt[0] <<" "<< mean_prt[transform_param_.crop_size()*transform_param_.crop_size()] << " "
      <<mean_prt[transform_param_.crop_size()*transform_param_.crop_size()*2];

    } else{
      for (int i = 0; i < datum_channels_*transform_param_.crop_size()*transform_param_.crop_size(); i++)
        mean_prt[i] = 128;
      LOG(INFO) << mean_prt[0] <<" "<< mean_prt[transform_param_.crop_size()*transform_param_.crop_size()] << " "
      <<mean_prt[transform_param_.crop_size()*transform_param_.crop_size()*2];
  }
  }
  mean_ = data_mean_.cpu_data();
  data_transformer_.InitRand();
  LOG(INFO) << "BaseDataLayer<Dtype>::LayerSetUp";
}

template <typename Dtype>
void BasePrefetchingDataLayer<Dtype>::LayerSetUp(
    const vector<Blob<Dtype>*>& bottom, vector<Blob<Dtype>*>* top) {
  BaseDataLayer<Dtype>::LayerSetUp(bottom, top);
  // Now, start the prefetch thread. Before calling prefetch, we make two
  // cpu_data calls so that the prefetch thread does not accidentally make
  // simultaneous cudaMalloc calls when the main thread is running. In some
  // GPUs this seems to cause failures if we do not so.
  this->prefetch_data_.mutable_cpu_data();
  if (this->output_labels_) {
    this->prefetch_label_.mutable_cpu_data();
  }
  DLOG(INFO) << "Initializing prefetch";
  this->CreatePrefetchThread();
  DLOG(INFO) << "Prefetch initialized.";
}

template <typename Dtype>
void BasePrefetchingDataLayer<Dtype>::CreatePrefetchThread() {
  this->phase_ = Caffe::phase();
  this->data_transformer_.phase_ = Caffe::phase();
  this->data_transformer_.InitRand();
  CHECK(StartInternalThread()) << "Thread execution failed";
}

template <typename Dtype>
void BasePrefetchingDataLayer<Dtype>::JoinPrefetchThread() {
  CHECK(WaitForInternalThreadToExit()) << "Thread joining failed";
}

template <typename Dtype>
void BasePrefetchingDataLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom, vector<Blob<Dtype>*>* top) {
  // First, join the thread
  JoinPrefetchThread();
  // Copy the data
  int skip = 1;
  caffe_copy(prefetch_data_.count(), prefetch_data_.cpu_data(),
             (*top)[0]->mutable_cpu_data());
  if (this->output_labels_) {
    caffe_copy(prefetch_label_.count(), prefetch_label_.cpu_data(),
               (*top)[1]->mutable_cpu_data());
    skip = 2;
  }
  for (int i = 0; i < top->size() - skip; i++){
  	  caffe_copy(prefetch_aux_data_[i]->count(), prefetch_aux_data_[i]->cpu_data(),
  			  (*top)[i+skip]->mutable_cpu_data());
  }
  // Start a new prefetch thread
  CreatePrefetchThread();
}

#ifdef CPU_ONLY
STUB_GPU_FORWARD(BasePrefetchingDataLayer, Forward);
#endif

INSTANTIATE_CLASS(BaseDataLayer);
INSTANTIATE_CLASS(BasePrefetchingDataLayer);

}  // namespace caffe
