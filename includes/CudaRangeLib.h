#include <vector>
#include <cstdint>

#ifndef ROS_WORLD_TO_GRID_CONVERSION
#define ROS_WORLD_TO_GRID_CONVERSION 1
#endif

#ifndef M_PI
#define M_PI 3.141592653589793238462643383279502
#endif

class RayMarchingCUDA
{
public:
	RayMarchingCUDA(std::vector<std::vector<float> > grid, int w, int h, float mr);
	~RayMarchingCUDA();
	void calc_range_many(float *ins, float *outs, int num_casts);
	void numpy_calc_range(float *ins, float *outs, int num_casts);
	void numpy_calc_range_angles(float * ins, float * angles, float * outs, int num_particles, int num_angles);
	void calc_range_repeat_angles_eval_sensor_model(float * ins, float * angles, float * obs, double * weights, int num_particles, int num_angles);

	void set_sensor_table(double *sensor_table, int table_width);

	#if ROS_WORLD_TO_GRID_CONVERSION == 1
	void set_conversion_params(float w_scale, float w_angle, float w_origin_x, 
		float w_origin_y, float w_sin_angle, float w_cos_angle) {
			inv_world_scale = 1.0 / w_scale; 
			world_scale = w_scale; 
			world_angle = w_angle;
			world_origin_x = w_origin_x;
			world_origin_y = w_origin_y;
			world_sin_angle = w_sin_angle;
			world_cos_angle = w_cos_angle;
			rotation_const = -1.0 * w_angle - 3.0*M_PI / 2.0;
			constants_set = true;
	}
	#endif
	
private:
	float *d_ins;
	float *d_outs;
	float *d_distMap;
	double *d_sensorTable;
	double *d_weights;
	int width;
	int height;
	int table_width;
	float max_range;

	bool allocated_weights = false;

	#if ROS_WORLD_TO_GRID_CONVERSION == 1
	float inv_world_scale;
	float world_scale;
	float world_angle;
	float world_origin_x;
	float world_origin_y;
	float world_sin_angle;
	float world_cos_angle;
	float rotation_const;
	bool constants_set = false;
	#endif
};

// GPU-backed GiantLUTCast. Additive companion to RayMarchingCUDA above -- same overall shape
// (device query/output buffers, world->grid conversion params, a single batched
// numpy_calc_range_angles entry point), but instead of ray-marching the distance transform it
// does one O(1) flat gather into a precomputed giant lookup table uploaded once at construction.
// The table is a contiguous row-major uint16_t buffer laid out as
// (x*height + y)*theta_discretization + theta_bin (built host-side in GiantLUTCastGPU, RangeLib.h).
// This is the "can GPU concurrency hide the LUT's DRAM latency once actually parallelized?"
// experiment -- deliberately a different, cheaper kernel than RayMarchingCUDA's variable-length
// dependent march. Purely additive: no existing class/kernel here is touched.
class GiantLUTCastCUDA
{
public:
	// flat_lut: contiguous host buffer of w*h*td uint16_t values (see layout note above).
	// max_div_limits: max_range / uint16_max, applied on lookup to rescale the quantized value
	// back to a range (mirrors GiantLUTCast::calc_range under _GIANT_LUT_SHORT_DATATYPE).
	GiantLUTCastCUDA(uint16_t *flat_lut, int w, int h, int td, float mr, float max_div_limits);
	~GiantLUTCastCUDA();

	// One kernel launch, processes exactly num_particles*num_angles queries. Does NOT auto-split
	// internally (that avoids the ceil-based chunking overflow found in RayMarchingGPU); any
	// chunking to keep a call under CHUNK_SIZE is the caller's responsibility (mcl_bench_lutgpu.cpp).
	void numpy_calc_range_angles(float * ins, float * angles, float * outs, int num_particles, int num_angles);

	#if ROS_WORLD_TO_GRID_CONVERSION == 1
	void set_conversion_params(float w_scale, float w_angle, float w_origin_x,
		float w_origin_y, float w_sin_angle, float w_cos_angle) {
			inv_world_scale = 1.0 / w_scale;
			world_scale = w_scale;
			world_angle = w_angle;
			world_origin_x = w_origin_x;
			world_origin_y = w_origin_y;
			world_sin_angle = w_sin_angle;
			world_cos_angle = w_cos_angle;
			rotation_const = -1.0 * w_angle - 3.0*M_PI / 2.0;
			constants_set = true;
	}
	#endif

private:
	float *d_ins;
	float *d_outs;
	uint16_t *d_lut;
	int width;
	int height;
	int theta_discretization;
	float max_range;
	float max_div_limits;

	#if ROS_WORLD_TO_GRID_CONVERSION == 1
	float inv_world_scale;
	float world_scale;
	float world_angle;
	float world_origin_x;
	float world_origin_y;
	float world_sin_angle;
	float world_cos_angle;
	float rotation_const;
	bool constants_set = false;
	#endif
};