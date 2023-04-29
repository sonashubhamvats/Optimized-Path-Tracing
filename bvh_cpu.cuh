#ifndef BVH_CPU_H
#define BVH_CPU_H

#include "aabb.cuh"
#include "bvh_gpu.cuh"
#include <vector>
#include <algorithm>
#include <queue>


class bvh_cpu{
    public:
        int start_index,end_index;
        aabb box;
        int index_bvh_arr;

        bvh_cpu();
        bvh_cpu(int s,int e,int i):
        start_index(s),end_index(e),index_bvh_arr(i){}

        
};

class bvh_cpu_sah{
    public:
        int start_index,end_index;
        aabb box;
        int index_bvh_arr;
        int next_left_empty_space_in_bvh_arr;
        int next_right_empty_space_in_bvh_arr;

        bvh_cpu_sah();
        bvh_cpu_sah(int s,int e,int i):
        start_index(s),end_index(e),index_bvh_arr(i){
            next_left_empty_space_in_bvh_arr = 1;
            next_right_empty_space_in_bvh_arr = 2;
        }
};

void form_gpu_bvh(aabb *list_of_bounding_boxes,bvh_gpu_node *b_arr,
bvh_cpu top_bvh_node)
{

    std::queue<std::pair<bvh_cpu,
    std::pair<int,int>>> q;

    
    q.push({top_bvh_node,{top_bvh_node.start_index,top_bvh_node.end_index}});

    while(!q.empty())
    {
        std::pair<bvh_cpu,
        std::pair<int,int>> top = q.front();
        q.pop();
        
        
        bvh_cpu bvh_cpu_node_top = top.first;
        int s_index = top.second.first;
        int l_index = top.second.second;

        
        
        bvh_cpu_node_top.box = list_of_bounding_boxes[s_index]; 
        
        for(int i=s_index+1;i<l_index;i++)
        {
            bvh_cpu_node_top.box = surrounding_box(bvh_cpu_node_top.box,list_of_bounding_boxes[i],
            -1);
            
        }
        
        int longest_axis = return_longest_axis(bvh_cpu_node_top.box);
        
        b_arr[bvh_cpu_node_top.index_bvh_arr] = bvh_gpu_node(bvh_cpu_node_top.box.index,
        bvh_cpu_node_top.index_bvh_arr,bvh_cpu_node_top.box);
        
        
        

        if(l_index-s_index>1)
        {
            int mid = (s_index+l_index)/2;


            std::nth_element(list_of_bounding_boxes+s_index,
            list_of_bounding_boxes+mid,list_of_bounding_boxes+l_index,
            [longest_axis](const aabb a, const aabb b) {
                return a.min()[longest_axis] < b.min()[longest_axis];
            });

            bvh_cpu left_node = bvh_cpu(s_index,mid,2*bvh_cpu_node_top.index_bvh_arr+1);
            bvh_cpu right_node = bvh_cpu(mid,l_index,2*bvh_cpu_node_top.index_bvh_arr+2);

        
            q.push({left_node,{s_index,mid}});
            q.push({right_node,{mid,l_index}});

        }

        
    }

    
}

void form_gpu_bvh_sah(aabb *list_of_bounding_boxes,bvh_gpu_node_sah *b_arr
,bvh_cpu_sah top_node_bvh_cpu)
{

    std::queue<std::pair<bvh_cpu_sah,
    std::pair<int,int>>> q;

    q.push({top_node_bvh_cpu,{top_node_bvh_cpu.start_index,top_node_bvh_cpu.end_index}});

    int nxt_left_empty_space_in_bvh_arr = top_node_bvh_cpu.next_left_empty_space_in_bvh_arr;
    int nxt_right_empty_space_in_bvh_arr = top_node_bvh_cpu.next_right_empty_space_in_bvh_arr;
    while(!q.empty())
    {
        std::pair<bvh_cpu_sah,
        std::pair<int,int>> top = q.front();
        q.pop();
        
        
        bvh_cpu_sah bvh_cpu_node_top = top.first;
        int s_index = top.second.first;
        int l_index = top.second.second;

        bvh_cpu_node_top.box = list_of_bounding_boxes[s_index];

        if((l_index-s_index)<=1)
        {
            b_arr[bvh_cpu_node_top.index_bvh_arr] = bvh_gpu_node_sah(bvh_cpu_node_top.box.index,
            bvh_cpu_node_top.index_bvh_arr,-1,-1,bvh_cpu_node_top.box);
            continue;
        }   
        
        for(int i=s_index+1;i<l_index;i++)
        {
            bvh_cpu_node_top.box = surrounding_box(bvh_cpu_node_top.box,list_of_bounding_boxes[i],-1);
        }

        int longest_axis_here = return_longest_axis(bvh_cpu_node_top.box);
        
        aabb l_box,r_box;
        bool l_box_initialized = false;
        bool r_box_initialized = false;
        int left_count = 0,best_left_count=0;
        int right_count = 0,best_right_count=0;
        float best_pos;
        int best_axis;
        float best_cost = FLT_MAX;


        int i = longest_axis_here;
        float bound_min  = bvh_cpu_node_top.box.min()[i];
        float bound_max = bvh_cpu_node_top.box.max()[i];

        float scale = (bound_max - bound_min) / 100;
        for(int j=1;j<100;j++)
        {
            float best_position = bound_min + j * scale;
            
            float cost_here = 0.0f;
            l_box_initialized = false;
            r_box_initialized = false;
            left_count = 0;
            right_count = 0;

            for(int k=s_index;k<l_index;k++)
            {
                if(list_of_bounding_boxes[k].get_centroid()[i]<best_position)
                {
                    left_count++;
                    if(!l_box_initialized)
                    {
                        l_box = list_of_bounding_boxes[k];
                        l_box_initialized=!l_box_initialized;
                    }
                    else
                    {
                        l_box = surrounding_box(l_box,list_of_bounding_boxes[k]);
                    }
                }
                else
                {
                    right_count++;
                    if(!r_box_initialized)
                    {
                        r_box = list_of_bounding_boxes[k];
                        r_box_initialized=!r_box_initialized;
                    }
                    else
                    {
                        r_box = surrounding_box(r_box,list_of_bounding_boxes[k]);
                    }
                }
            }
            cost_here = l_box.get_area()*left_count + r_box.get_area()*right_count;
            
            if(cost_here<best_cost)
            {
                best_cost = cost_here;
                best_axis = i;
                best_pos = best_position;
                best_left_count = left_count;
                best_right_count = right_count;
            }
        }

        
        
        float split_pos = best_pos;


        i = s_index;
        int j = l_index-1;
        

        if((l_index-s_index)*bvh_cpu_node_top.box.get_area()<=best_cost)
        {
            int mid = (l_index+s_index)/2;
            best_left_count = (l_index-s_index)/2;
            std::nth_element(list_of_bounding_boxes+s_index,
            list_of_bounding_boxes+mid,list_of_bounding_boxes+l_index,
            [best_axis](aabb a, aabb b) {
                return a.min()[best_axis]<b.min()[best_axis];
            });
        }
        else
        {
            while(i<=j)
            {
                if(list_of_bounding_boxes[i].get_centroid()[best_axis]<split_pos)
                {
                    i++;
                }       
                else
                {
                    aabb box  = list_of_bounding_boxes[i];
                    list_of_bounding_boxes[i] = list_of_bounding_boxes[j];
                    list_of_bounding_boxes[j--] = box;
                }             
            }
        }
    
        
        b_arr[bvh_cpu_node_top.index_bvh_arr] = bvh_gpu_node_sah(bvh_cpu_node_top.box.index
        ,bvh_cpu_node_top.index_bvh_arr,nxt_left_empty_space_in_bvh_arr,nxt_right_empty_space_in_bvh_arr,bvh_cpu_node_top.box);

        

        
        bvh_cpu_sah left_node = bvh_cpu_sah(s_index,s_index+best_left_count,nxt_left_empty_space_in_bvh_arr);
        bvh_cpu_sah right_node = bvh_cpu_sah(s_index+best_left_count,l_index,nxt_right_empty_space_in_bvh_arr);

        nxt_left_empty_space_in_bvh_arr=nxt_right_empty_space_in_bvh_arr+1;
        nxt_right_empty_space_in_bvh_arr = nxt_left_empty_space_in_bvh_arr+1;


        q.push({left_node,{s_index,s_index+best_left_count}});
        q.push({right_node,{s_index+best_left_count,l_index}});
    }
    
    
}

#endif