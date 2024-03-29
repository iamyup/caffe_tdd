// Copyright 2014 BVLC and contributors.

#include <vector>

#include "caffe/layer.hpp"
#include "caffe/vision_layers.hpp"
#include "caffe/util/im2col_old.hpp"
#include "caffe/filler.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void TiledConvolutionLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top) {
  kernel_size_ = this->layer_param_.tiled_convolution_param().kernel_size();
  stride_ = this->layer_param_.tiled_convolution_param().stride();
  group_ = this->layer_param_.tiled_convolution_param().group();
  pad_ = this->layer_param_.tiled_convolution_param().pad();
  num_ = bottom[0]->num();
  channels_ = bottom[0]->channels();
  height_ = bottom[0]->height();
  width_ = bottom[0]->width();
  NTILE_WIDTH_ = this->layer_param_.tiled_convolution_param().ntile_width();
  NTILE_HEIGHT_ = this->layer_param_.tiled_convolution_param().ntile_height();
  // TODO: generalize to handle inputs of different shapes.
  for (int bottom_id = 1; bottom_id < bottom.size(); ++bottom_id) {
    CHECK_EQ(num_, bottom[bottom_id]->num()) << "Inputs must have same num.";
    CHECK_EQ(channels_, bottom[bottom_id]->channels())
        << "Inputs must have same channels.";
    CHECK_EQ(height_, bottom[bottom_id]->height())
        << "Inputs must have same height.";
    CHECK_EQ(width_, bottom[bottom_id]->width())
        << "Inputs must have same width.";
  }
  num_output_ = this->layer_param_.tiled_convolution_param().num_output();
  CHECK_GT(num_output_, 0);
  CHECK_EQ(channels_ % group_, 0);
  // The im2col result buffer would only hold one image at a time to avoid
  // overly large memory usage.
  int height_out = (height_ + 2 * pad_ - kernel_size_) / stride_ + 1;
  int width_out = (width_ + 2 * pad_ - kernel_size_) / stride_ + 1;
  //col_buffer_.Reshape(
   //   1, channels_ * kernel_size_ * kernel_size_, height_out, width_out);

  CHECK(height_out % NTILE_HEIGHT_ == 0);
  CHECK(width_out % NTILE_WIDTH_ == 0);
  TILE_WIDTH_ = width_out / NTILE_WIDTH_;
  TILE_HEIGHT_ = height_out / NTILE_HEIGHT_;

  col_buffer_.Reshape(1, channels_ * kernel_size_ * kernel_size_ , TILE_HEIGHT_, TILE_WIDTH_);
  out_buffer_.Reshape(1, num_output_, TILE_HEIGHT_, TILE_WIDTH_);

  // Set the parameters
  CHECK_EQ(num_output_ % group_, 0)
      << "Number of output should be multiples of group.";
  bias_term_ = this->layer_param_.tiled_convolution_param().bias_term();
  // Figure out the dimensions for individual gemms.
  M_ = num_output_ / group_;
  K_ = channels_ * kernel_size_ * kernel_size_ / group_;
  N_ = TILE_WIDTH_ * TILE_HEIGHT_;
  for (int top_id = 0; top_id < top->size(); ++top_id) {
    (*top)[top_id]->Reshape(num_, num_output_, height_out, width_out);
  }
  int ntiles = NTILE_WIDTH_ * NTILE_HEIGHT_;
  // Check if we need to set up the weights
  if (this->blobs_.size() > 0) {
    LOG(INFO) << "Skipping parameter initialization";
  } else {
    if (bias_term_) {
      this->blobs_.resize(2*ntiles);
    } else {
      this->blobs_.resize(1*ntiles);
    }
    // Intialize the weight
    for(int i = 0; i < ntiles; i++) {
	    this->blobs_[i].reset(
		new Blob<Dtype>(num_output_, channels_/ group_, kernel_size_, kernel_size_));
	    // fill the weights
	    shared_ptr<Filler<Dtype> > weight_filler(
		GetFiller<Dtype>(this->layer_param_.tiled_convolution_param().weight_filler()));
	    weight_filler->Fill(this->blobs_[i].get());
	    // If necessary, intiialize and fill the bias term
	    if (bias_term_) {
	      this->blobs_[ntiles+i].reset(new Blob<Dtype>(1, 1, 1, num_output_));
	      shared_ptr<Filler<Dtype> > bias_filler(
		  GetFiller<Dtype>(this->layer_param_.tiled_convolution_param().bias_filler()));
	      bias_filler->Fill(this->blobs_[ntiles+i].get());
	    }

    }
  }
  // Set up the bias filler
  if (bias_term_) {
    //bias_multiplier_.reset(new SyncedMemory(N_ * sizeof(Dtype)));
    bias_multiplier_.Reshape(1, 1, 1, N_);
    Dtype* bias_multiplier_data =
        reinterpret_cast<Dtype*>(bias_multiplier_.mutable_cpu_data());
    for (int i = 0; i < N_; ++i) {
        bias_multiplier_data[i] = 1.;
    }
  }
  // Propagate gradients to the parameters (as directed by backward pass).
  //this->param_propagate_down_.resize(this->blobs_.size(), true);
}

// !!!!!!!!!!!!!!!!!!!!!!!!!!
// !!!! to be modified
template <typename Dtype>
void TiledConvolutionLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top) {
//  num_ = bottom[0]->num();
//  height_ = bottom[0]->height();
//  width_ = bottom[0]->width();
//  LOG(INFO) << channels_ << " " << bottom[0]->channels();
//  CHECK_EQ(bottom[0]->channels(), channels_) << "Input size incompatible with"
//    " tiled convolution kernel.";
//  // TODO: generalize to handle inputs of different shapes.
//  for (int bottom_id = 1; bottom_id < bottom.size(); ++bottom_id) {
//    CHECK_EQ(num_, bottom[bottom_id]->num()) << "Inputs must have same num.";
//    CHECK_EQ(channels_, bottom[bottom_id]->channels())
//        << "Inputs must have same channels.";
//    CHECK_EQ(height_, bottom[bottom_id]->height())
//        << "Inputs must have same height.";
//    CHECK_EQ(width_, bottom[bottom_id]->width())
//        << "Inputs must have same width.";
//  }
//  // Shape the tops.
//  height_out_ = (height_ + 2 * pad_ - kernel_size_) / stride_ + 1;
//  width_out_ = (width_ + 2 * pad_ - kernel_size_) / stride_ + 1;
//  for (int top_id = 0; top_id < top->size(); ++top_id) {
//    (*top)[top_id]->Reshape(num_, num_output_, height_out_, width_out_);
//  }
//  // Prepare the matrix multiplication computation.
//  // Each input will be convolved as a single GEMM.
//  M_ = num_output_ / group_;
//  K_ = channels_ * kernel_size_ * kernel_size_ / group_;
//  N_ = height_out_ * width_out_;
//  // The im2col result buffer will only hold one image at a time to avoid
//  // overly large memory usage.
//  col_buffer_.Reshape(
//      1, channels_ * kernel_size_ * kernel_size_, height_out_, width_out_);
//  for (int top_id = 0; top_id < top->size(); ++top_id) {
//    (*top)[top_id]->Reshape(num_, num_output_, height_out_, width_out_);
//  }
//  // Set up the all ones "bias multiplier" for adding biases by BLAS
//  if (bias_term_) {
//    bias_multiplier_.Reshape(1, 1, 1, N_);
//    caffe_set(N_, Dtype(1), bias_multiplier_.mutable_cpu_data());
//  }
}

template <typename Dtype>
void TiledConvolutionLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top) {
  for (int i = 0; i < bottom.size(); ++i) {
    const Dtype* bottom_data = bottom[i]->cpu_data();
    Dtype* top_data = (*top)[i]->mutable_cpu_data();
    Dtype* col_data = col_buffer_.mutable_cpu_data();
    const Dtype* weight = this->blobs_[0]->cpu_data();
    int weight_offset = M_ * K_;
    int col_offset = K_ * N_;
    int top_offset = M_ * N_;
    for (int n = 0; n < num_; ++n) {
      // First, im2col
      im2col_cpu(bottom_data + bottom[i]->offset(n), channels_, height_,
                        width_, kernel_size_, pad_, stride_, col_data);
      // Second, innerproduct with groups
      for (int g = 0; g < group_; ++g) {
        caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, M_, N_, K_,
          (Dtype)1., weight + weight_offset * g, col_data + col_offset * g,
          (Dtype)0., top_data + (*top)[i]->offset(n) + top_offset * g);
      }
      // third, add bias
      if (bias_term_) {
//        caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num_output_,
//            N_, 1, (Dtype)1., this->blobs_[1]->cpu_data(),
//            reinterpret_cast<const Dtype*>(bias_multiplier_->cpu_data()),
//            (Dtype)1., top_data + (*top)[i]->offset(n));
          caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num_output_,
              N_, 1, (Dtype)1., this->blobs_[1]->cpu_data(),
              reinterpret_cast<const Dtype*>(bias_multiplier_.cpu_data()),
              (Dtype)1., top_data + (*top)[i]->offset(n));
      }
    }
  }
  //return Dtype(0.);
}

template <typename Dtype>
void TiledConvolutionLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, vector<Blob<Dtype>*>* bottom) {
  const Dtype* weight = this->blobs_[0]->cpu_data();
  Dtype* weight_diff = this->blobs_[0]->mutable_cpu_diff();
  caffe_set(this->blobs_[0]->count(), Dtype(0), weight_diff);
  Dtype* bias_diff = NULL;
  if (bias_term_) {
    bias_diff = this->blobs_[1]->mutable_cpu_diff();
    caffe_set(this->blobs_[1]->count(), Dtype(0), bias_diff);
  }
  const int weight_offset = M_ * K_;
  const int col_offset = K_ * N_;
  const int top_offset = M_ * N_;
  for (int i = 0; i < top.size(); ++i) {
    const Dtype* top_diff = top[i]->cpu_diff();
    const Dtype* bottom_data = (*bottom)[i]->cpu_data();
    Dtype* bottom_diff = (*bottom)[i]->mutable_cpu_diff();
    Dtype* col_data = col_buffer_.mutable_cpu_data();
    Dtype* col_diff = col_buffer_.mutable_cpu_diff();

    // Bias gradient, if necessary.
    if (bias_term_) {
      for (int n = 0; n < num_; ++n) {
//        caffe_cpu_gemv<Dtype>(CblasNoTrans, num_output_, N_,
//            1., top_diff + top[0]->offset(n),
//            static_cast<const Dtype*>(bias_multiplier_->cpu_data()), 1.,
//            bias_diff);
          caffe_cpu_gemv<Dtype>(CblasNoTrans, num_output_, N_,
              1., top_diff + top[0]->offset(n),
              static_cast<const Dtype*>(bias_multiplier_.cpu_data()), 1.,
              bias_diff);
      }
    }
    for (int n = 0; n < num_; ++n) {
      // Since we saved memory in the forward pass by not storing all col data,
      // we will need to recompute them.
      im2col_cpu(bottom_data + (*bottom)[i]->offset(n), channels_, height_,
                 width_, kernel_size_, pad_, stride_, col_data);
      // gradient w.r.t. weight. Note that we will accumulate diffs.
      for (int g = 0; g < group_; ++g) {
        caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasTrans, M_, K_, N_,
          (Dtype)1., top_diff + top[i]->offset(n) + top_offset * g,
          col_data + col_offset * g, (Dtype)1.,
          weight_diff + weight_offset * g);
      }
      // gradient w.r.t. bottom data, if necessary
      if (propagate_down[i]) {
        for (int g = 0; g < group_; ++g) {
          caffe_cpu_gemm<Dtype>(CblasTrans, CblasNoTrans, K_, N_, M_,
            (Dtype)1., weight + weight_offset * g,
            top_diff + top[i]->offset(n) + top_offset * g,
            (Dtype)0., col_diff + col_offset * g);
        }
        // col2im back to the data
        col2im_cpu(col_diff, channels_, height_, width_, kernel_size_, pad_,
            stride_, bottom_diff + (*bottom)[i]->offset(n));
      }
    }
  }
}

INSTANTIATE_CLASS(TiledConvolutionLayer);

}  // namespace caffe
