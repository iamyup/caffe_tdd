#include <fstream>
#include <iostream>
#include <string>
#include <utility>
#include <vector>

#include "caffe/data_layers.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/rng.hpp"

#ifdef USE_MPI
#include "mpi.h"
#include <boost/filesystem.hpp>
using namespace boost::filesystem;
#endif

namespace caffe{
template <typename Dtype>
VideoDataLayer<Dtype>:: ~VideoDataLayer<Dtype>(){
	this->JoinPrefetchThread();
}

template <typename Dtype>
void VideoDataLayer<Dtype>:: DataLayerSetUp(const vector<Blob<Dtype>*>& bottom, vector<Blob<Dtype>*>* top){
	const int new_height  = this->layer_param_.video_data_param().new_height();
	const int new_width  = this->layer_param_.video_data_param().new_width();
	const int new_length  = this->layer_param_.video_data_param().new_length();
	const string& source = this->layer_param_.video_data_param().source();
	LOG(INFO) << "Opening file" << source;
	std:: ifstream infile(source.c_str());
	string filename;
	int label;
	int length;
	while (infile >> filename >> length >> label){
		lines_.push_back(std::make_pair(filename,label));
		lines_duration_.push_back(length);
	}
	if (this->layer_param_.video_data_param().shuffle()){
		const unsigned int prefectch_rng_seed = caffe_rng_rand();
		prefetch_rng_1_.reset(new Caffe::RNG(prefectch_rng_seed));
		prefetch_rng_2_.reset(new Caffe::RNG(prefectch_rng_seed));
		ShuffleVideos();
	}

	LOG(INFO) << "A total of " << lines_.size() << "videos.";
	lines_id_ = 0;

	Datum datum;
	const unsigned int frame_prefectch_rng_seed = caffe_rng_rand();
	frame_prefetch_rng_.reset(new Caffe::RNG(frame_prefectch_rng_seed));
	caffe::rng_t* frame_rng = static_cast<caffe::rng_t*>(frame_prefetch_rng_->generator());
	int offset = (*frame_rng)() % (lines_duration_[lines_id_] - new_length + 1);
	CHECK(ReadFlowToDatum(lines_[lines_id_].first, lines_[lines_id_].second, offset, new_height, new_width, new_length, &datum));
	LOG(INFO) << "LENGTH:" << new_length;

	const int crop_size = this->layer_param_.transform_param().crop_size();
	const int batch_size = this->layer_param_.video_data_param().batch_size();
	if (crop_size > 0){
		(*top)[0]->Reshape(batch_size, datum.channels(), crop_size, crop_size);
		this->prefetch_data_.Reshape(batch_size, datum.channels(), crop_size, crop_size);
	} else {
		(*top)[0]->Reshape(batch_size, datum.channels(), datum.height(), datum.width());
		 this->prefetch_data_.Reshape(batch_size, datum.channels(), datum.height(), datum.width());
	}
	LOG(INFO) << "output data size: " << (*top)[0]->num() << "," << (*top)[0]->channels() << "," << (*top)[0]->height() << "," << (*top)[0]->width();

	(*top)[1]->Reshape(batch_size, 1, 1, 1);
	this->prefetch_label_.Reshape(batch_size, 1, 1, 1);

	// datum size
	this->datum_channels_ = datum.channels();
	this->datum_height_ = datum.height();
	this->datum_width_ = datum.width();
	this->datum_size_ = datum.channels() * datum.height() * datum.width();
	this->phase_ = Caffe::phase();
	this->data_transformer_.phase_ = Caffe::phase();
}

template <typename Dtype>
void VideoDataLayer<Dtype>::ShuffleVideos(){
	caffe::rng_t* prefetch_rng1 = static_cast<caffe::rng_t*>(prefetch_rng_1_->generator());
	caffe::rng_t* prefetch_rng2 = static_cast<caffe::rng_t*>(prefetch_rng_2_->generator());
	shuffle(lines_.begin(), lines_.end(), prefetch_rng);
	shuffle(lines_duration_.begin(), lines_duration_.end(),prefetch_rng);
}

template <typename Dtype>
void VideoDataLayer<Dtype>::InternalThreadEntry(){
	Datum datum;
	CHECK(this->prefetch_data_.count());
	Dtype* top_data = this->prefetch_data_.mutable_cpu_data();
	Dtype* top_label = this->prefetch_label_.mutable_cpu_data();
	VideoDataParameter video_data_param = this->layer_param_.video_data_param();
	const int batch_size = video_data_param.batch_size();
	const int new_height = video_data_param.new_height();
	const int new_width = video_data_param.new_width();
	const int new_length = video_data_param.new_length();
	const int lines_size = lines_.size();

	#ifndef USE_MPI
		for (int item_id = 0; item_id < batch_size; ++item_id){
	#else
		for (int item_id = batch_size* Caffe::mpi_self_rank()*(-1);
				item_id < batch_size*(Caffe::mpi_all_rank() - Caffe::mpi_self_rank()); ++item_id){
		bool do_read = (item_id>=0) && (item_id<batch_size);
		if (do_read){
	#endif
		CHECK_GT(lines_size, lines_id_);
		int offset = 0;
		if (this->phase_ == Caffe::TRAIN){
			caffe::rng_t* rng = static_cast<caffe::rng_t*>(frame_prefetch_rng_->generator());
			int offset = (*rng)() % (lines_duration_[lines_id_]-new_length+1);
		}
		else{
			offset = (int) (lines_duration_[lines_id_]-new_length+1)/2
		}
		if(!ReadFlowToDatum(lines_[lines_id_].first, lines_[lines_id_].second, offset, new_height, new_width, new_length, &datum)) {
			continue;
		}
		this->data_transformer_.Transform(item_id, datum, this->mean_, top_data);
		top_label[item_id] = datum.label();
 	#ifdef USE_MPI
		}
	#endif
		//next iteration
		lines_id_++;
		if (lines_id_ >= lines_size) {
			DLOG(INFO) << "Restarting data prefetching from start.";
			lines_id_ = 0;
			if(this->layer_param_.video_data_param().shuffle()){
				ShuffleVideos();
			}
		}

	}

}

INSTANTIATE_CLASS(VideoDataLayer);
}

