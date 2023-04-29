#include <iostream>

int main(){
    int total_no_objects[] = {10,100,1000,10000};
    int t_size = 4;
    int nx_1 = 512 , nx_2 = 736 , nx_3 = 1280;
    int ny_1 = 448 , ny_2 = 640 , ny_3 = 992;
    for(int i=4;i<=21;i++)
    {
        if(i!=18)
        {
            for(int k=0;k<t_size-1;k++)
            {
                std::cout<<"./exefiles/master_renderer_gpu -"<<i<<" -"
                <<total_no_objects[k]<<" -"<<nx_1<<" -"<<ny_1<<" -bvh >>./output/output.txt\n";
                std::cout<<"./exefiles/master_renderer_gpu -"<<i<<" -"
                <<total_no_objects[k]<<" -"<<nx_2<<" -"<<ny_2<<" -bvh >>./output/output.txt\n";
                std::cout<<"./exefiles/master_renderer_gpu -"<<i<<" -"
                <<total_no_objects[k]<<" -"<<nx_3<<" -"<<ny_3<<" -bvh >>./output/output.txt\n";
            }
            std::cout<<"read -p "<<"Press enter to continue\n";
            for(int k=t_size-1;k<t_size;k++)
            {
                std::cout<<"./exefiles/master_renderer_gpu -"<<i<<" -"
                <<total_no_objects[k]<<" -"<<nx_1<<" -"<<ny_1<<" -bvh >>./output/output.txt\n";
                std::cout<<"read -p "<<"Press enter to continue\n";
                std::cout<<"./exefiles/master_renderer_gpu -"<<i<<" -"
                <<total_no_objects[k]<<" -"<<nx_2<<" -"<<ny_2<<" -bvh >>./output/output.txt\n";
                std::cout<<"read -p "<<"Press enter to continue\n";
                std::cout<<"./exefiles/master_renderer_gpu -"<<i<<" -"
                <<total_no_objects[k]<<" -"<<nx_3<<" -"<<ny_3<<" -bvh >>./output/output.txt\n";
                std::cout<<"read -p "<<"Press enter to continue\n";
            }
        }

        if((i)%3==0)
            std::cout<<"read -p "<<"Press enter to continue\n";
        
        
    }
}