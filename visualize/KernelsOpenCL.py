#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 28 11:21:40 2017

@author: diegothomas
"""
Kernel_Test = """
__kernel void Test(__global float *TSDF) {

        int x = get_global_id(0); /*height*/
        int y = get_global_id(1); /*width*/
        int z = get_global_id(2); /*depth*/
        TSDF[x + 512*y + 512*512*z] = 1.0f;
}
"""
#__global float *prevTSDF, __global float *Weight
#__read_only image2d_t VMap

Kernel_FuseTSDF ="""
__kernel void FuseSortedTSDF( __global short int *TSDF,__global short int *Weight, __global short int *Dist, __constant short int *Depth,  
                           __constant float *VoxelVertexArray, const int nu, \
                           __constant float *intrinsic, const int n_row, const int m_col) {
              
        int voxid = get_global_id(0);
        int id = 0;
        int idx = 0;
        float convVal = 32767.0f; //Bu uint16 en max degeri. Eger pixel imageda degilse en yuksek degeri atiyoruz TSDF tableda
        for (id = 0; id < 4; id ++){
            idx = (voxid *12 )+ 3 * id;  

            

            float4 pt;
            int2 pix;

            pt.x = VoxelVertexArray[idx+0] ; 
            pt.y = VoxelVertexArray[idx+1] ; 
            pt.z = VoxelVertexArray[idx+2] ; 

            //pt.x = SortedVertexArray[summitid*3+0] ;//VertexArray[(VoxelArray[idx])*3+0] ;
            //pt.y = SortedVertexArray[summitid*3+1] ;
            //pt.z = SortedVertexArray[summitid*3+2] ;
            
            
            pix.x = convert_int(round((pt.x/fabs( pt.z ))*intrinsic[0] + intrinsic[2]) );
            pix.y = convert_int(round((pt.y/fabs( pt.z ))*intrinsic[4] + intrinsic[5]) );
            //short int pixx = convert_int(round(((pt.x)/fabs( pt.z ))*intrinsic[0] + intrinsic[2] ));
            //short int pixy = convert_int(round(((pt.y)/fabs( pt.z ))*intrinsic[4] + intrinsic[5] ));
            
            
            //Dist[summitid] = pt.x ;//convert_int(round(VertexArray[idx] *1000.0));//convert_int(round(pt.x*1000.0));
            //Check if the pixel inside the frame 
            if( pix.x < 0|| pix.x > (m_col-1) || pix.y < 0 || pix.y > (n_row-1))
            {
                    TSDF[(voxid*4)+id] = (short int)(convVal);
                    //Dist[idx] =  0.0;
            }

            else 
            { //Pixel in the frame 
                //Compute distance between project voxel and surface in the spehere equation
                
                int index = pix.x + (m_col)*pix.y;
                if(Depth[index] == 0)
                {
                    TSDF[(voxid*4)+id] = (short int)(convVal);
                    

                }
                else{
                    //Dist[summitid] = 1.0;
                    pt.z = (float)(pt.z) * nu; //numVal bu numero dogru mu bak aslnda 
                    pt.z = (short int)(round(pt.z));
                    //short int dist = -(pt.z - Depth[index]+150.0);
                    short int dist = -(pt.z - (Depth[index]/1000.));
                    Dist[(voxid*4)+id] =  index;//(short int)round((dist));
                    if (dist > 1.0f) dist = 1.0f;
                    if (dist < -1.0f) dist = -1.0f;
                    //else dist = max(-1.0f, dist);

                    float prev_tsdf = ((float)TSDF[(voxid*4)+id])/convVal;
                    float prev_weight = (float)(Weight[(voxid*4)+id]);
                    float w = 1.0f;

                    
                    TSDF[(voxid*4)+id] =  (short int)(round(((prev_tsdf*prev_weight+dist*w)/(prev_weight+w))*convVal));
                    //TSDF[summitid] =  (short int)(round((dist*convVal)));
                    //pt.x;//pix.x;//Depth[index];// Depth[index];//(tdist*w)/(2.0);
                    if (dist >=0)
                        Weight[(voxid*4)+id] = min(1000, Weight[(voxid*4)+id]+1);
                    
                                     
         
                }



            }



        }

        
         

}
"""
Kernel_FuseEqTSDF = """
__kernel void FuseEqTSDF(__global short int *TSDF, __constant short int *TSDFtable, __constant int *Dim,
                           __constant float *Pose, __constant int *VoxelArray, __constant float *VertexArray, const int n_row, const int m_col) {

        int vertid = get_global_id(0); // 0 to 3 
        int voxid = get_global_id(1); // 0 to 333
        
        int idx = 4*voxid+vertid;
        float convVal = 32767.0f;
        float4 pt;
        int2 pix;
        pt.x = VertexArray[(VoxelArray[voxid*4+vertid])*3+0] ;
        pt.y = VertexArray[(VoxelArray[voxid*4+vertid])*3+1] ;
        pt.z = VertexArray[(VoxelArray[voxid*4+vertid])*3+2] ;

        //pix.x = convert_int(round((pt.x/fabs( (pt.z)) ) ));
        //pix.y = convert_int(round((pt.y/fabs( (pt.z)) ) ));

        //Check if the pixel inside the frame 
        //if (pix.x < 0 || pix.x > m_col-1 || pix.y < 0 || pix.y > n_row-1)
        //            TSDF[idx] = convVal;
        
        //else{ //Pixel in the frame 
            //Compute distance between project voxel and surface in the spehere equation
            //sphere equation (x-a)*(x-a) + (y-b)*(y-b) + (z-c)*(z-c) = r*r 
            //ag = pow(mag, 0.5f);
            float dist = (pt.x-1.5)*(pt.x-1.5) + (pt.y-1.5)*(pt.y-1.5) + (pt.z-1.5)*(pt.z-1.5) ;
            dist = pow(dist,0.5f); 
            //float dist = pt.x + pt.y + pt.z;
            //dist = (short int) dist ;
            //if dist < 0.5 it is inside the sphere. 0

            if(dist < 0.4)
                TSDF[idx] = (dist - 0.4 )/ 0.4;
                //TSDF[idx] = -1;
            //if distance is 0.5 it is on the sphere
            else if(dist == 0.4)
                TSDF[idx] = 0;
            //Else, outlier 
            else if(dist > 0.4)
                TSDF[idx] = (dist - 0.4 )/ 0.44;
                //TSDF[idx] = 1;
            //TSDF[idx] = TSDFtable[idx] ;
        //}
        //TSDF[idx] =TSDFtable[idx];
  
}


"""