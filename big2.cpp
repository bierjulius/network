#include <stdlib.h>
#include <iostream>
#include <iomanip>
#include <fstream>
#include <math.h>
#include <time.h>
#include <algorithm>
using namespace std;

#define img_count  6
#define img_size  4
#define ans_size  2
#define internal_size  8
#define inhib_count  2
#define max_fire  2
#define probability 40
#define min_seed 0
#define seed_count 1000
#define updates_without_var 20
#define updates_with_var 10
#define neutral_failure 500
#define sample_size 30 //Vielfaches von img_count
#define randomizer 10000

const int response_size=8;
const int response[response_size] = {0,0,0,1,2,2,2,2};
const int images[img_count][img_size+ans_size]={
		{1,1,0,0, 0,0},
		{0,0,1,1, 0,0},
		{0,1,0,1, 0,1},
		{1,0,1,0, 0,1},
		{0,1,1,0, 1,0},
		{1,0,0,1, 1,0}
	};

const int hop_size = img_count*img_size;
const int net_size = hop_size+internal_size+img_size+2*ans_size;
const int max_line_count=internal_size*(img_size+2*ans_size+internal_size-1);

int inputs[img_count][hop_size]={
	{1,1,0,0, 0,0,0,0, 0,0,0,0, 0,0,0,0, 0,0,0,0, 0,0,0,0,},
	{0,0,0,0, 0,0,1,1, 0,0,0,0, 0,0,0,0, 0,0,0,0, 0,0,0,0,},
	{0,0,0,0, 0,0,0,0, 0,1,0,1, 0,0,0,0, 0,0,0,0, 0,0,0,0,},
	{0,0,0,0, 0,0,0,0, 0,0,0,0, 1,0,1,0, 0,0,0,0, 0,0,0,0,},
	{0,0,0,0, 0,0,0,0, 0,0,0,0, 0,0,0,0, 0,1,1,0, 0,0,0,0,},
	{0,0,0,0, 0,0,0,0, 0,0,0,0, 0,0,0,0, 0,0,0,0, 1,0,0,1,}
};

int dummy[hop_size]={-1};
void update(int[hop_size]=dummy,bool=false);
int variance(int[ans_size]);
void print_status();
float actual_fitness();

int neurons[net_size];
int img_prob[img_count];
int adjacency[net_size][net_size];
int inhib[net_size];
int order[max_line_count];
int line_array[max_line_count][2];
float saved_fitness[seed_count];


int temp_vector[net_size];


int main(){
	int time_start=time(NULL);
	ofstream outfile( "./"+to_string(time_start)+"_Seeds.txt");
	outfile << "img_count: " << img_count;
	outfile << "\nimg_size: " << img_size;
	outfile << "\nans_size: "<<ans_size;
	outfile << "\ninternal_size: " << internal_size;
	outfile << "\ninhib_count: "<<inhib_count;
	outfile<<"\nmax_fire: "<<max_fire;
	outfile<<"\nprobability: "<<probability;
	outfile << "\nSeeds: "<<min_seed<<"-"<<min_seed+seed_count-1;
	outfile<<"\nupdates_without_var: "<<updates_without_var;
	outfile<<"\nupdates_with_var: "<<updates_with_var;
	outfile<<"\nneutral_failure: "<<neutral_failure;
	outfile<<"\nsample_size: "<<sample_size;
	outfile<<"\nrandomizer: "<<randomizer;


	//Definiton of the (inhibitory) factor for each neuron
	for(int i=0;i<net_size-inhib_count;i++){
		inhib[i]=1;
	}
	for(int i=net_size-inhib_count;i<net_size;i++){
		inhib[i]=-1;
	}

	//Definition of the line_array. Each possible line has a number up to max_line_count.
	int lc=0;

	//Connections from recurrent input (Hopfield output) to internal neurons
	for(int i=hop_size+ans_size;i<hop_size+ans_size+img_size;i++){
		for(int j=hop_size+ans_size+img_size;j<net_size-ans_size;j++){
			line_array[lc][0]=i;
			line_array[lc][1]=j;
			lc++;
		}
	}
	//Connections from internal neurons to output neurons (and vice versa)
	for(int i=hop_size+ans_size+img_size;i<net_size-ans_size;i++){
		for(int j=net_size-ans_size;j<net_size;j++){
			line_array[lc][0]=i;
			line_array[lc][1]=j;
			lc++;
			line_array[lc][0]=j;
			line_array[lc][1]=i;
			lc++;
		}
	}
	//Connections between internal neurons
	for(int i=hop_size+ans_size+img_size;i<net_size-ans_size;i++){
		for(int j=hop_size+ans_size+img_size;j<i;j++){
			line_array[lc][0]=i;
			line_array[lc][1]=j;
			lc++;
			line_array[lc][0]=j;
			line_array[lc][1]=i;
			lc++;
		}
	}

    //Definition of the adjacency matrix
	for(int i=0;i<net_size;++i){
		for(int j=0;j<net_size;++j){
			adjacency[i][j]=0;
		}
	}
	//Fixed connections in the Hopfield net
		for(int k=0;k<hop_size;k+=img_size){
			for(int i=0;i<img_size;i++){
				for(int j=i+1;j<img_size;j++){
					adjacency[i+k][j+k]
					=adjacency[j+k][i+k]
					=images[k/img_size][i]*images[k/img_size][j];

				}
				//Fixed connection to the Hopfield answers
				for(int j=0;j<ans_size;j++){
					adjacency[j+hop_size][i+k]=images[k/img_size][i]*images[k/img_size][img_size+j];
				}
				//Fixed connection to the recurrent input
				for(int j=0;j<img_size;j++){
					adjacency[j+hop_size+ans_size][i+k]=int(i==j);
				}

			}
		}
	//Searching for perfect learners with given Seeds
	for(int iseed=min_seed;iseed<min_seed+seed_count;iseed++){
		int prev_time=time(NULL);
		srand(iseed);
		//Random connections in the recurrent net
		for(int i=0;i<max_line_count;++i){
			if(rand()%100<probability){
				adjacency[line_array[i][1]][line_array[i][0]]=1;
			} else{
				adjacency[line_array[i][1]][line_array[i][0]]=0;
			}
		}

		float fitness=1;
		int ntest=0;
		while(ntest<neutral_failure){
			ntest++;
			int best_line=-1;
			float best_var=fitness;
			for (int i=0;i<max_line_count;i++){
				order[i]=i;
			}
			random_shuffle(begin(order),end(order));

			for (int l=0;l<max_line_count;l++){
				adjacency[line_array[order[l]][1]][line_array[order[l]][0]]=1-adjacency[line_array[order[l]][1]][line_array[order[l]][0]];
				float var=actual_fitness();


				adjacency[line_array[order[l]][1]][line_array[order[l]][0]]=1-adjacency[line_array[order[l]][1]][line_array[order[l]][0]];	
				if (var<best_var){
					best_line=order[l];
					best_var=var;
					ntest=0;
					break;
				} else if (best_line<0&&var==best_var){
					best_line=order[l];
				}
			}
			if (best_line<0){
				break;
			}
			adjacency[line_array[best_line][1]][line_array[best_line][0]]=1-adjacency[line_array[best_line][1]][line_array[best_line][0]];			
			fitness=best_var;
		}
		cout << "\nSeed: "<<iseed<<" | Time: "<<time(NULL)-prev_time<<" | Fitness: "<<fitness;
		outfile<< "\nSeed: "<<iseed<<" | Fitness: "<<fitness;
		if (fitness==0){
			outfile<< "<---------| Actual Fitness: "<<actual_fitness();
			cout << "<----------|";
		}
		saved_fitness[iseed-min_seed]=fitness;
	}
	float avg_best_fitness;
	for(int i=0;i<seed_count;i++){
		avg_best_fitness+=saved_fitness[i];
	}
	avg_best_fitness=avg_best_fitness/seed_count;
	outfile <<"\n\nAverage best fitness: "<<avg_best_fitness;
	outfile <<"\nAverage calculaton time: "<<float(time(NULL)-time_start)/seed_count<<"s";
}
void update(int input[hop_size],bool update_hop){
	if(update_hop){
		if (input[0]>=0){
			for(int i=0;i<hop_size;i++){
				neurons[i]=input[i];
			}
		}
		for(int i=0;i<hop_size;i++){
			int sum=0;
			for(int j=0;j<hop_size;j++){
				sum+=adjacency[i][j]*neurons[j];
			}
			if (sum==0){
				temp_vector[i]=0;
			} else {
				temp_vector[i]=1;
			}
		}
		for(int i=hop_size;i<hop_size+ans_size+img_size;i++){
			int sum=0;
			for(int j=0;j<hop_size;j++){
				sum+=adjacency[i][j]*neurons[j];
			}
			if (sum==0){
				temp_vector[i]=0;
			} else {
				temp_vector[i]=max_fire;
			}
		}
	}
	for(int i=hop_size+img_size+ans_size;i<net_size;i++){
		int sum=0;
		for(int j=hop_size+ans_size;j<net_size;j++){
			sum+=adjacency[i][j]*neurons[j]*inhib[j];
		}
		if (sum<=0){
			temp_vector[i]=0;
		} else if (sum<response_size) {
			temp_vector[i]=response[sum];
		} else if(sum>response_size) {
			temp_vector[i]=max_fire;
		}
	}
	for(int i=0;i<net_size;++i){
		neurons[i]=temp_vector[i];
	}
}

int variance(int desired[ans_size]){
	int err=0;
	int var=0;
	for(int i=0;i<ans_size;i++){
		err =neurons[net_size-ans_size+i]-desired[i];
		var += err*err;
	}
	return var;
}

void print_status(){
	cout<<"\nHop Output/Rec Input: ";
	for (int i=hop_size+ans_size;i<hop_size+ans_size+img_size;i++){
		cout << neurons[i];
	}
	cout<<"\nHop Answer: ";
	for (int i=hop_size;i<hop_size+ans_size;i++){
		cout << neurons[i];
	}
	cout<<"\nRec internal: ";
	for (int i=hop_size+ans_size+img_size;i<net_size-ans_size;i++){
		cout << neurons[i];
	}
	cout<<" | Rec Answer: ";
	for (int i=net_size-ans_size;i<net_size;i++){
		cout << neurons[i];
	}
	cout<<" | Variance: "<< variance(&neurons[hop_size]);
	cin.get();
}

float actual_fitness(){
	 int var=0;
	 for(int k=0;k<3;k++){
		 for (int i=0;i<img_count;i++){
		 	if(rand()%randomizer){
				update(inputs[i],true);
				update(inputs[i],true);
			} else{
				int random_image=rand()%img_count;
				update(inputs[random_image],true);
				update(inputs[random_image],true);
			}
			for(int j=0;j<updates_without_var;++j){
				update();					
			}
			// Updates mit Vergleich mit Responsvektor
			for(int j=0;j<updates_with_var;++j){
				update();
				var+=variance(&neurons[hop_size]);
			}

		}
	}
	 float adjusted_fitness=var;
	 adjusted_fitness=adjusted_fitness/(3*ans_size*img_count*max_fire*max_fire*updates_with_var);
	 return adjusted_fitness;
}
