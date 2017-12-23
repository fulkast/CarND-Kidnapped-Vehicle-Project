/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 */

#include <algorithm>
#include <iostream>
#include <numeric>
#include <climits>
#include <math.h>
#include <iostream>
#include <sstream>
#include <string>
#include <iterator>

#include "particle_filter.h"

using namespace std;

double eval2DGaussian(double x_diff, double y_diff, double std_x, double std_y){

	double dev_x2 =  x_diff * x_diff;
	double dev_y2 =  y_diff * y_diff;
	double var_x = std_x * std_x;
	double var_y = std_y * std_y;
	double normalizer = 1./ ( 2.* M_PI * std_x * std_y);

	double x_contrib = dev_x2 / var_x;
 	double y_contrib = dev_y2 / var_y;

	double exponent = -0.5 * (x_contrib + y_contrib);
	// std::cout << "exponent " << exponent << std::endl;
	return normalizer * exp(exponent);
}

void ParticleFilter::init(double x, double y, double theta, double std[]) {
	// TODO: Set the number of particles. Initialize all particles to first position (based on estimates of
	//   x, y, theta and their uncertainties from GPS) and all weights to 1.
	// Add random Gaussian noise to each particle.
	// NOTE: Consult particle_filter.h for more information about this method (and others in this file).

	// Initialize number of particles
	num_particles = 50;

	// Initialize random generator
	double std_x, std_y, std_theta; // Standard deviations for x, y, and theta
	std_x = std[0];
	std_y = std[1];
	std_theta = std[2];
	// This line creates a normal (Gaussian) distribution for x.
	normal_distribution<double> dist_x(x, std_x);
	// TODO: Create normal distributions for y and theta.
	normal_distribution<double> dist_y(y, std_y);
	normal_distribution<double> dist_theta(theta, std_theta);

	for (int i = 0; i < num_particles; i++) {
		particles.push_back(Particle());
		double sample_x, sample_y, sample_theta;
		sample_x = dist_x(gen_);
		sample_y = dist_y(gen_);
		sample_theta = dist_theta(gen_);
		particles[i].x = sample_x;
		particles[i].y = sample_y;
		particles[i].theta = sample_theta;
		particles[i].weight = 1.;
		weights.push_back(1.);
		//cout << "Sample " << i + 1 << " " << particles[i].x << " " << particles[i].y <<
		// " " << particles[i].theta << endl;
	}

	cumulative_weights.resize(weights.size());

	is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
	// TODO: Add measurements to each particle and add random Gaussian noise.
	// NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
	//  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
	//  http://www.cplusplus.com/reference/random/default_random_engine/

	// std::cout << "\ndelta t, velocity, yaw_rate " << delta_t << " "
	// 	<< velocity << " " << yaw_rate << std::endl;

	// Initialize random generator
	double std_x, std_y, std_theta; // Standard deviations for x, y, and theta
	double var_scale = 1.;
	std_x = std_pos[0] * var_scale;
	std_y = std_pos[1] * var_scale;
	std_theta = std_pos[2] * var_scale;

	// This line creates a normal (Gaussian) distribution for x.
	normal_distribution<double> dist_x(0, std_x);
	normal_distribution<double> dist_y(0, std_y);
	normal_distribution<double> dist_theta(0, std_theta);
	double sample_x, sample_y, sample_theta;

	for (int i = 0; i < num_particles; i++) {
		double x,y,theta;
		x = particles[i].x;
		y = particles[i].y;
		theta = particles[i].theta;
		if (fabs(yaw_rate) > 0.001) {
			x += velocity / yaw_rate * (sin(theta + yaw_rate * delta_t) - sin(theta));
			y += velocity / yaw_rate * (cos(theta) - cos(theta + yaw_rate * delta_t));
		} else {
			x += velocity * cos(theta) * delta_t;
			y += velocity * sin(theta) * delta_t;
		}
		theta += yaw_rate * delta_t;

		// get perturbances
		sample_x = dist_x(gen_);
		sample_y = dist_y(gen_);
		sample_theta = dist_theta(gen_);

		x += sample_x;
		y += sample_y;
		theta += sample_theta;

		particles[i].x = x;
		particles[i].y = y;
		particles[i].theta = theta;

	}

}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
	// TODO: Find the predicted measurement that is closest to each observed measurement and assign the
	//   observed measurement to this particular landmark.
	// NOTE: this method will NOT be called by the grading code. But you will probably find it useful to
	//   implement this method and use it as a helper during the updateWeights phase.

	// Tactic: O(mn) implementation. For each predicted landmark iterate through
	// all observations to find the closest observation
	for (int i = 0; i < observations.size(); i++) {
		double obs_x, obs_y, min_dist;
		int best_id = -1;
		obs_x = observations[i].x;
		obs_y = observations[i].y;
		min_dist = std::numeric_limits<double>::max();
		for (int j = 0; j < predicted.size(); j++) {
			double dist = sqrt((obs_x - predicted[j].x) * (obs_x - predicted[j].x)
												+ (obs_y - predicted[j].y) * (obs_y - predicted[j].y));
			if (dist < min_dist) {
				best_id = j;
				min_dist = dist;
			}
		}
		// std::cout << "best id " << best_id << " min_dist " << min_dist << std::endl;
		observations[i].id = best_id;
	}

}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[],
		const std::vector<LandmarkObs> &observations, const Map &map_landmarks) {
	// TODO: Update the weights of each particle using a mult-variate Gaussian distribution. You can read
	//   more about this distribution here: https://en.wikipedia.org/wiki/Multivariate_normal_distribution
	// NOTE: The observations are given in the VEHICLE'S coordinate system. Your particles are located
	//   according to the MAP'S coordinate system. You will need to transform between the two systems.
	//   Keep in mind that this transformation requires both rotation AND translation (but no scaling).
	//   The following is a good resource for the theory:
	//   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
	//   and the following is a good resource for the actual equation to implement (look at equation
	//   3.33
	//   http://planning.cs.uiuc.edu/node99.html

	double var_scale = 1.;
	std_landmark[0] *= var_scale;
	std_landmark[1] *= var_scale;

	std::vector<LandmarkObs> permanent_landmarks(map_landmarks.landmark_list.size());
	for (int i = 0; i < permanent_landmarks.size(); i++) {
		permanent_landmarks[i].id = map_landmarks.landmark_list[i].id_i;
		permanent_landmarks[i].x = map_landmarks.landmark_list[i].x_f;
		permanent_landmarks[i].y = map_landmarks.landmark_list[i].y_f;
	}

	// transform the observations according to the pose of each particle
	for (int i = 0; i < num_particles; i++) {

		// std::cout << "particle weight before: " << particles[i].weight << std::endl;

		std::vector<LandmarkObs> transformed_observations(observations.size());
		double particle_x, particle_y, particle_theta;
		particle_x = particles[i].x;
		particle_y = particles[i].y;
		particle_theta = particles[i].theta;

		// std::cout << "particle " << i << " x: " << particle_x << " y: " <<
		// particle_y << " theta: " << particle_theta << " observations: " << std::endl;

		for (int j = 0; j < transformed_observations.size(); j++) {
			double observation_x, observation_y, tmp_x, tmp_y;
			observation_x = observations[j].x;
			observation_y = observations[j].y;
			tmp_x = particle_x + cos(particle_theta) * observation_x
																					- sin(particle_theta) * observation_y;
			tmp_y = particle_y + sin(particle_theta) * observation_x +
			 																		+ cos(particle_theta) * observation_y;
 			transformed_observations[j].x = tmp_x;
			transformed_observations[j].y = tmp_y;
			// std::cout << "observation " << j << "x: " << tmp_x << "y: " << tmp_y << std::endl;
		}

		// match the proposed observations to landmarks
		dataAssociation(permanent_landmarks, transformed_observations);

		// calculate the likelihood
		double particle_weight = 1.;
		particles[i].associations.clear();
		particles[i].sense_x.clear();
		particles[i].sense_y.clear();
		for (auto landmark : transformed_observations) {
			// set associations
			particles[i].associations.push_back(permanent_landmarks[landmark.id].id);
			particles[i].sense_x.push_back(landmark.x);
			particles[i].sense_y.push_back(landmark.y);

			// std::cout << "xdiff " << landmark.x - permanent_landmarks[landmark.id].x <<
			//  " std_x " << std_landmark[0] <<  std::endl;
			// std::cout << "x_diff: " << landmark.x - permanent_landmarks[landmark.id].x << std::endl;

			double score = eval2DGaussian(landmark.x - permanent_landmarks[landmark.id].x,
														landmark.y - permanent_landmarks[landmark.id].y,
													std_landmark[0], std_landmark[1]);
      // std::cout << "score: " << score << std::endl;
			particle_weight *= score;
			// std::cout << "score " << score << std::endl;
		}
		// std::cout << i << "th particle first observation associated landmark " <<
		// 			transformed_observations[0].id << " weight " << particle_weight << std::endl;

		particles[i].weight = particle_weight;
		if (particle_weight == 0.) {
			// std::cout << "dieing particle x,y,theta: " << particles[i].x << " "
			// << particles[i].y << " " << particles[i].theta  << std::endl;
		}

	}

	double weight_total = 0.;
	double min_weight = std::numeric_limits<double>::max();
	double max_weight = -1;
	int best_particle = -1;

	for (auto particle : particles) {
		if (particle.weight < min_weight && particle.weight > 0)
			min_weight = particle.weight;
	}

	if (weight_total == 0.) {
		for (auto particle : particles) {
			particle.weight = 1.;
		}
	}

	if (min_weight == std::numeric_limits<double>::max()) exit(0);

	// std::cout << "min weight " << min_weight << std::endl;

	for (auto particle : particles)  weight_total += particle.weight;
	// std::cout << "total_weight " << weight_total << std::endl;
	int index = 0;
	for (auto &particle : particles) {
		particle.weight /= weight_total;
		if (particle.weight > max_weight) {
			best_particle = index;
			max_weight = particle.weight;
		}
		weights[index++] = particle.weight;

		// std::cout << "particle " << index << " weight " << particle.weight << std::endl;
	}

	// std::cout << "\nbest particle x,y,theta: " << particles[best_particle].x << " "
	// 	<< particles[best_particle].y << " " << particles[best_particle].theta << std::endl;

	// weight_total = 0;
	// for (auto particle : particles)  weight_total += particle.weight;
	// std::cout << "total_weight after normalization: " << weight_total << std::endl;

  std::partial_sum(weights.begin(), weights.end(),
	 							cumulative_weights.begin());
  // for (auto cumsum : cumulative_weights) {
	// 	std::cout << cumsum << " ";
	// }
	// std::cout << std::endl;
}

void ParticleFilter::resample() {
	// TODO: Resample particles with replacement with probability proportional to their weight.
	// NOTE: You may find std::discrete_distribution helpful here.
	//   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution

	std::random_device rd;  //Will be used to obtain a seed for the random number engine
  std::mt19937 gen(rd()); //Standard mersenne_twister_engine seeded with rd()
  std::uniform_real_distribution<> dis(0.0, 1.0);

	std::vector<Particle> new_particles;
	new_particles.reserve(num_particles);

	for (int i = 0; i < num_particles; i++) {
		double random_weight = dis(gen);
		int index = 0;
		while (cumulative_weights[index] < random_weight &&
																	index < num_particles-1) index++;
		new_particles.push_back(particles[index]);
	}

	particles = new_particles;

	// std::cout << "\n\n";

}

Particle ParticleFilter::SetAssociations(Particle& particle, const std::vector<int>& associations,
                                     const std::vector<double>& sense_x, const std::vector<double>& sense_y)
{
    //particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
    // associations: The landmark id that goes along with each listed association
    // sense_x: the associations x mapping already converted to world coordinates
    // sense_y: the associations y mapping already converted to world coordinates

    particle.associations= associations;
    particle.sense_x = sense_x;
    particle.sense_y = sense_y;
}

string ParticleFilter::getAssociations(Particle best)
{
	vector<int> v = best.associations;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<int>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseX(Particle best)
{
	vector<double> v = best.sense_x;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseY(Particle best)
{
	vector<double> v = best.sense_y;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
