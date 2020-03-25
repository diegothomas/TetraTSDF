import imp
import numpy as np
import pyopencl as cl
import math
import time
from math import sqrt
PI = math.pi
GPU = imp.load_source('GPUManager', './GPUManager.py')
KernelsOpenCL = imp.load_source('KernelsOpenCL', './KernelsOpenCL.py')
#General = imp.load_source('General', './lib/General.py')

mf = cl.mem_flags

savepath = "output/debug"

class MarchingTetrahedra():
    
    def __init__(self, Size, TSDF,Iso,voxelvertexArray, voxelArray, vertexArray, GPUManager):
        return
        """
        self.MT = My_MT.MarchingTetrahedra(self.TSDFManager.Size, 0.0, voxelArray,vertexArray self.GPUManager)

        Constructor
        :param Size: Size of the volume
        :param Res: resolution parameters
        :param Iso: isosurface to render
        :param GPUManager: GPU environment
        """
        self.Size = Size #TSDFManagerin sizei ise bu [1 4 333] TSDFTable kadar ise 1332
        self.iso = Iso
        self.count = 0 #count of faces after MT
        
        self.nb_faces = np.array([0])
        self.offset = np.array([0], dtype = np.int32)
        self.offsetarray = np.zeros((self.Size[2]))#, dtype = np.int32)
        self.nb_vertices = np.array([0], dtype = np.int32)
        #mantiken bir vozel max iki face yaratabiliyorsa, birazdan yaratacgimiz face counter da max voxcount x 2 kadar olabilir 
        #self.FaceVerticesArray -> runtwo algorithm. Double memory allocation

        self.VoxelArray = voxelArray #int
        self.VertexArray = vertexArray #float
        self.VertexArray = self.VertexArray.ravel()

        self.GPUManager = GPUManager
        self.TSDFTable = np.zeros((self.Size[2],self.Size[1]),dtype=np.float32)
        
        self.TSDFTable = self.TSDFTable.ravel() 

        #self.VertexArray = np.array([0.5,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,0.2,0.36,0.321456987])
        #self.VoxelArray = np.array([0,1,2,3,0,1,2,3])
        for it,tsdf in enumerate(TSDF):
            #if it < 4:
            self.TSDFTable[it] = round((float(tsdf)/32767.0),4)
        

        #self.TSDFTable =TSDF
        print("TSDFTAble size check")
        print(self.TSDFTable.shape )
        np.savetxt(savepath + '/tsdfnedir.txt', self.TSDFTable, delimiter=',')
        np.savetxt(savepath + '/voxelArraynedir.txt', self.VoxelArray, delimiter=',')
        np.savetxt(savepath + '/vertexArraynedir.txt', self.VertexArray, delimiter=',')
        '''
        ucgencount = 0 
        for ind in range(0,self.Size[2]):
            idd=ind*4
            cc = 0
            if(self.TSDFTable[idd] == -1 ):
                cc+=1
            if(self.TSDFTable[idd+1] == -1 ):
                cc+=1  
            if(self.TSDFTable[idd+2] == -1 ):
                cc+=1 
            if(self.TSDFTable[idd+3] == -1 ) :
                cc+=1
            if(cc == 1 ) :
                ucgencount +=1
            elif(cc == 2) :
                ucgencount += 2
            elif(cc==3):
                ucgencount +=1
        print("KAC UCGEN OLMALI MARCHINGCUBE ")
        print(ucgencount)   
        ''' 
        #offset is for count number of faces in runone. Increment the offset when a new face is found.
        self.OffsetGPU = cl.Buffer(self.GPUManager.context, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf = self.offset)
        self.OffsetArrayGPU = cl.Buffer(self.GPUManager.context, mf.READ_WRITE, self.offsetarray.nbytes)      
        # Create the programs
        #For runtwopass
        #self.FaceCounterGPU = cl.Buffer(self.GPUManager.context, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf = self.nb_faces)
        #self.FaceCounterGPU =cl.Buffer(self.GPUManager.context, mf.READ_WRITE, self.nb_faces.nbytes)        
        #self.GPUManager.programs['MarchingCubesIndexing'] = cl.Program(self.GPUManager.context, KernelsOpenCL.Kernel_MarchingCubeIndexing).build()
        #self.GPUManager.programs['MarchingCubes'] = cl.Program(self.GPUManager.context, KernelsOpenCL.Kernel_MarchingCubesolo).build()
        #for runonepass
        self.FaceVerticesArray = np.zeros((6 *self.Size[2],3),dtype=np.float32)
        print("self.size")
        print(self.Size[2])
        print(" self.FaceVerticesArray::::::::::")
        print( self.FaceVerticesArray.shape)
        self.FaceVerticesArray = self.FaceVerticesArray.ravel() 
        self.FaceVerticesGPU =cl.Buffer(self.GPUManager.context, mf.READ_WRITE, self.FaceVerticesArray.nbytes)  
        print( self.FaceVerticesArray.shape)

        

        
        # Allocate arrays for GPU
        
        #self.IndexGPU = cl.Buffer(self.GPUManager.context, mf.READ_WRITE, tmp.nbytes)
        #su onemli 
        
        self.Size_Volume = cl.Buffer(self.GPUManager.context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf = Size)
        self.VertexGPU =cl.Buffer(self.GPUManager.context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf = self.VertexArray)
        self.VoxelGPU =cl.Buffer(self.GPUManager.context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf = self.VoxelArray)
        self.TSDFGPU =cl.Buffer(self.GPUManager.context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf = self.TSDFTable)
        self.Vertices = np.zeros((6*self.Size[2], 3), dtype = np.float32)
        self.Verticestwopass = np.zeros((3*self.Size[2], 3), dtype = np.float64)
        #below are for CPU Marching Tetrahedra
        self.aw = np.zeros((self.Size[2],self.Size[1]), dtype=np.float32)
        self.bw = np.zeros((self.Size[2],self.Size[1]), dtype=np.float32)
        self.cw = np.zeros((self.Size[2],self.Size[1]), dtype=np.float32)
        self.dw = np.zeros((self.Size[2],self.Size[1]), dtype=np.float32)
        self.aw = self.aw.ravel()
        self.bw = self.bw.ravel()
        self.cw = self.cw.ravel()
        self.dw = self.dw.ravel()


        self.voxelvertexArray = voxelvertexArray
        self.voxelvertexArray = self.voxelvertexArray.ravel()
        self.voxelvertexArrayGPU =cl.Buffer(self.GPUManager.context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf = self.voxelvertexArray)

        ############# tempolary comment out #############
        #self.GPUManager.programs['MarchingTetrahedra'] = cl.Program(self.GPUManager.context, KernelsOpenCL.Kernel_MarchingCube).build()
        #################################################
        #self.GPUManager.programs['MarchingCubesAndIndexing'] = cl.Program(self.GPUManager.context, KernelsOpenCL.Kernel_MarchingCube).build()

    def runtwopassGPU(self, VolGPU):
        """
        First find all the edges that are crossed by the surface
        Second compute face and vertices according to the edges
 
        """ 
      
        self.nb_faces[0] = 0
        self.faceCounts = np.zeros((self.Size[2]), dtype = np.int32)
        # Create Buffer for the number of faces
        #cl.enqueue_write_buffer(self.GPUManager.queue, self.FaceCounterGPU, self.nb_faces).wait()
        
        self.GPUManager.programs['MarchingCubesIndexing'].MarchingCubesIndexing(self.GPUManager.queue, (self.Size[2],self.Size[1]), None, \
                                self.TSDFGPU ,self.OffsetArrayGPU, np.int32(self.iso),self.FaceCounterGPU,self.Size_Volume).wait()

        # read buffer to know number of faces
        #cl.enqueue_read_buffer(self.GPUManager.queue, self.FaceCounterGPU, self.nb_faces).wait()

        #print "nb Faces: ", self.nb_faces[0]  / 4
        #self.count = self.nb_faces[0] / 4
        self.count = 0 
        cl.enqueue_read_buffer(self.GPUManager.queue, self.OffsetArrayGPU, self.faceCounts).wait()
        for i in range(1,self.Size[2]):
            self.count += self.faceCounts[i]
            self.faceCounts[i] += self.faceCounts [i-1]
        np.savetxt('OffsetArrayGPU.txt', self.faceCounts, delimiter=',')
        print("facecounts")
        print(self.count)  
        #self.count = self.faceCounts
        cl.enqueue_write_buffer(self.GPUManager.queue, self.OffsetArrayGPU, self.faceCounts).wait()
        # Memory allocations
        
        self.Verticestwopass = np.zeros((3*self.count, 3), dtype = np.float32) 
        self.Verticestwopass = self.Verticestwopass.ravel()
        self.Normales = np.zeros((self.count, 3), dtype = np.float32) 
        self.Normales = self.Normales.ravel()
        # Write numpy array into  buffer array
        self.VerticesGPU = cl.Buffer(self.GPUManager.context, mf.READ_WRITE, self.Verticestwopass.nbytes)
        #VertexGPU is the vertex values of the tetracube 
        #VerticesGPU is the vertex values of new triangles inside tetrahedrons of tetracube
        self.NormalesGPU = cl.Buffer(self.GPUManager.context, mf.READ_WRITE, self.Normales.nbytes)
        # Compute faces and vertices. 

        self.GPUManager.programs['MarchingCubes'].MarchingCubes(self.GPUManager.queue, (self.Size[2],self.count), None, \
                                self.TSDFGPU ,self.OffsetArrayGPU, self.VertexGPU,self.VoxelGPU,np.int32(self.iso), self.VerticesGPU, self.NormalesGPU, self.Size_Volume).wait()
        #cl.enqueue_write_buffer(self.GPUManager.queue, self.FaceCounterGPU, self.nb_faces).wait()
     
        # Read buffer array into numpy array
        cl.enqueue_read_buffer(self.GPUManager.queue, self.VerticesGPU, self.Verticestwopass).wait()
        #np.savetxt('Vertices_runtwopass_fromgpu.txt', self.Vertices, delimiter=',') 

        #TSDF DENEMEK icin okutsana  
        '''
        self.TSDFCheckValues = np.zeros((self.Size[1],self.Size[2]), dtype=np.int16) 
        self.TSDFCheckValues = self.TSDFCheckValues.ravel()
        cl.enqueue_read_buffer(self.GPUManager.queue, self.TSDFGPU, self.TSDFCheckValues).wait()
        np.savetxt('TSDFCheckValues.txt', self.TSDFCheckValues, delimiter=',') 
        '''
        #cl.enqueue_read_buffer(self.GPUManager.queue, self.NormalesGPU, self.Normales).wait()
        #np.savetxt('Normales_runtwopass.txt', self.Normales, delimiter=',')

        #Normalize
        #for i in range(0,self.count):
            #i *= 3
            #mag = sqrt((self.Normales[i]*self.Normales[i])+(self.Normales[i+1]*self.Normales[i+1])+(self.Normales[i+2]*self.Normales[i+2]))    
                  
            #self.Normales[i] =  self.Normales[i] / mag
            #self.Normales[i+1] =  self.Normales[i+1] / mag
            #self.Normales[i+2] =  self.Normales[i+2] / mag
            
        #np.savetxt('Normalized_runtwopass.txt', self.Normales, delimiter=',')
    def runonepassGPU(self, VolGPU):
        
        #self.Vertices = np.zeros((6*self.Size[2], 3), dtype = np.float64) #bunlari ravel 
        self.Vertices = self.Vertices.ravel() #672*9
        self.VoxelArray = self.VoxelArray.ravel()
        self.VertexArray = self.VertexArray.ravel()
        # #for CPU Calculation
        # self.MT()
        # return
        #for GPU calculation
        
        #print(self.TSDFGPU)
                
        #self.GPUManager.programs['MarchingCubesAndIndexing'].MarchingCubesAndIndexing(self.GPUManager.queue, (self.Size[2],self.Size[1]), None, \
        #                        self.TSDFGPU, self.VertexGPU,self.VoxelGPU,np.float32(self.iso), self.FaceVerticesGPU,  self.Size_Volume).wait()
        ######## tempolary comment out#####
        #self.GPUManager.programs['MarchingTetrahedra'].MarchingTetrahedra(self.GPUManager.queue, (self.Size[2],self.Size[1]), None, \
        #                        self.TSDFGPU,self.voxelvertexArrayGPU, self.VertexGPU,self.VoxelGPU,np.float32(self.iso), self.FaceVerticesGPU,  self.Size_Volume).wait()
        ###################################
        cl.enqueue_read_buffer(self.GPUManager.queue, self.FaceVerticesGPU, self.Vertices).wait()
        np.savetxt(savepath + '/FaceVerticesGPU.txt', self.Vertices, delimiter=',')
        #np.savetxt('CPUCheck.txt', self.Vertices, delimiter=',')
        
        #np.savetxt('selfaw.txt',self.aw, delimiter=',')
        #np.savetxt('selfbw.txt',self.bw, delimiter=',')
        #np.savetxt('selfcw.txt',self.cw, delimiter=',')
        #np.savetxt('selfdw.txt',self.dw, delimiter=',')
        
        count = 0
        voxcount = self.Size[2]
        print("check voxcount")
        print(voxcount)
        for cc in range (0,voxcount*2):
            c = cc * 9
            if not ((self.Vertices[c] == 0.0) and (self.Vertices[c+1] == 0.0) and (self.Vertices[c+2] == 0.0) \
                and (self.Vertices[c+3] == 0.0) and (self.Vertices[c+4] == 0.0) and (self.Vertices[c+5] == 0.0) \
                and (self.Vertices[c+6] == 0.0) and (self.Vertices[c+7] == 0.0) and (self.Vertices[c+8] == 0.0) ):
                count += 1 

        print("Count after MT :  ")
        print(count)
        self.count = count 
        #print "number of FaceVertices after : ", self.Vertices.Shape


    def run_CPU(self, TSDF, iso, voxelArray, vertexArray):
        
        self.Vertices_CPU = []
        voxcount = len(voxelArray)
        convVal = 32767.0
        for voxids in voxelArray:

            # Load 4 summits
            a = vertexArray[voxids[0]]
            b = vertexArray[voxids[1]]
            c = vertexArray[voxids[2]]
            d = vertexArray[voxids[3]]
            
            # Load 4 TSDF values
            aw = float(TSDF[voxids[0]])/ convVal
            bw = float(TSDF[voxids[1]])/ convVal
            cw = float(TSDF[voxids[2]])/ convVal
            dw = float(TSDF[voxids[3]])/ convVal

            # Check summits 
            count =0
            if(aw >= iso):
                count +=1
            if(bw >= iso):
                count +=1
            if(cw >=iso):
                count +=1
            if(dw >= iso):
                count +=1
            # if no faces
            if (count==0 or count==4):
                continue

            # if(abs(aw)==1 and abs(bw)==1 and abs(bw)==1 and abs(bw)==1):
            #     continue

            # three vertices are outside
            elif (count == 3):
                if (dw < iso): 
                    sort_a = a
                    sort_aw = aw
                    sort_b = b
                    sort_bw = bw
                    sort_c = c
                    sort_cw = cw
                    sort_d = d
                    sort_dw = dw
                
                elif (cw < iso):
                    sort_a = a
                    sort_aw = aw 
                    sort_b = d
                    sort_bw = dw
                    sort_c = b
                    sort_cw = bw
                    sort_d = c
                    sort_dw = cw
                
                elif (bw  < iso): 
                    sort_a = a
                    sort_aw = aw
                    sort_b = c
                    sort_bw = cw
                    sort_c = d
                    sort_cw = dw
                    sort_d = b
                    sort_dw = bw
                
                else :
                    sort_a = b
                    sort_aw = bw
                    sort_b = d
                    sort_bw = dw
                    sort_c = c
                    sort_cw = cw
                    sort_d = a
                    sort_dw = aw
    
                sort_aw = 1.0/(0.00001 + abs(sort_aw))
                sort_bw = 1.0/(0.00001 + abs(sort_bw))
                sort_cw = 1.0/(0.00001 + abs(sort_cw))
                sort_dw = 1.0/(0.00001 + abs(sort_dw))

                #bu degil
                da_frac = (sort_aw) / (sort_aw + sort_dw)
                ad_frac = 1.0-da_frac
                #da_frac = (sort_dw) / (sort_aw + sort_dw)
                db_frac = (sort_bw) / (sort_bw + sort_dw)
                bd_frac = 1.0-db_frac
                #db_frac = (sort_dw) / (sort_bw + sort_dw)
                dc_frac = (sort_cw) / (sort_cw + sort_dw)
                cd_frac = 1.0-dc_frac
                #dc_frac = (sort_dw) / (sort_cw + sort_dw)
                           
                self.Vertices_CPU += [da_frac*(sort_a) + (ad_frac)*sort_d]
                self.Vertices_CPU += [dc_frac*(sort_c) + (cd_frac)*sort_d]
                self.Vertices_CPU += [db_frac*(sort_b) + (bd_frac)*sort_d]


            # two polygons
            elif (count == 2) :
                if (aw >= iso and bw >= iso) :
                    sort_a = a
                    sort_aw = aw
                    sort_b = b
                    sort_bw = bw
                    sort_c = c
                    sort_cw = cw
                    sort_d = d
                    sort_dw = dw
                
                elif (aw >= iso and cw >= iso) :
                    sort_a = a
                    sort_aw = aw
                    sort_b = c
                    sort_bw = cw
                    sort_c = d
                    sort_cw = dw
                    sort_d = b
                    sort_dw = bw
                elif (aw >= iso and dw >= iso) :                
                    sort_a = a
                    sort_aw = aw 
                    sort_b = d
                    sort_bw = dw
                    sort_c = b
                    sort_cw = bw
                    sort_d = c
                    sort_dw = cw
                
                elif (bw >= iso and cw >= iso) :
                    sort_a = b
                    sort_aw = bw
                    sort_b = c
                    sort_bw = cw
                    sort_c = a
                    sort_cw = aw
                    sort_d = d
                    sort_dw = dw
                
                elif (bw >= iso and dw >= iso) :
                    sort_a = b
                    sort_aw = bw
                    sort_b = d
                    sort_bw = dw
                    sort_c = c
                    sort_cw = cw
                    sort_d = a
                    sort_dw = aw

                else :
                    sort_a = c
                    sort_aw = cw
                    sort_b = d
                    sort_bw = dw
                    sort_c = a
                    sort_cw = aw
                    sort_d = b
                    sort_dw = bw
            
                
                sort_aw = 1.0/(0.00001 + abs(sort_aw))
                sort_bw = 1.0/(0.00001 + abs(sort_bw))
                sort_cw = 1.0/(0.00001 + abs(sort_cw))
                sort_dw = 1.0/(0.00001 + abs(sort_dw))
                
                ac_frac = (sort_cw) / (sort_aw + sort_cw)
                ca_frac = 1.0-ac_frac
                #ca_frac = (sort_aw) / (sort_aw + sort_cw)
                ad_frac = (sort_dw) / (sort_aw + sort_dw)
                da_frac = 1.0-ad_frac
                #da_frac = (sort_aw) / (sort_aw + sort_dw)
                bc_frac = (sort_cw) / (sort_bw + sort_cw)
                cb_frac = 1.0-bc_frac
                #cb_frac = (sort_bw) / (sort_bw + sort_cw)
                bd_frac = (sort_dw) / (sort_bw + sort_dw)
                db_frac = 1.0-bd_frac
                #db_frac = (sort_bw) / (sort_bw + sort_dw)

                ad = da_frac*(sort_a) + (ad_frac)*sort_d
                bd = db_frac*(sort_b) + (bd_frac)*sort_d
                ac = ca_frac*(sort_a) + (ac_frac)*sort_c
                bc = cb_frac*(sort_b) + (bc_frac)*sort_c
                self.Vertices_CPU += [ac]
                self.Vertices_CPU += [bc]
                self.Vertices_CPU += [ad]
                self.Vertices_CPU += [bc]
                self.Vertices_CPU += [bd]
                self.Vertices_CPU += [ad]
                
            elif (count == 1) :
                if (aw >= iso):
                    sort_a = a
                    sort_aw = aw
                    sort_b = b
                    sort_bw = bw
                    sort_c = c
                    sort_cw = cw
                    sort_d = d
                    sort_dw = dw
                
                elif (bw >= iso):
                    sort_a = b
                    sort_aw = bw
                    sort_b = c
                    sort_bw = cw
                    sort_c = a
                    sort_cw = aw
                    sort_d = d
                    sort_dw = dw
                
                elif (cw >= iso) :
                    sort_a = c
                    sort_aw = cw
                    sort_b = a
                    sort_bw = aw
                    sort_c = b
                    sort_cw = bw
                    sort_d = d
                    sort_dw = dw
                
                else :
                    sort_a = d
                    sort_aw = dw
                    sort_b = c
                    sort_bw = cw
                    sort_c = b
                    sort_cw = bw
                    sort_d = a
                    sort_dw = aw
                                
                sort_aw = 1.0/(0.00001 + abs(sort_aw))
                sort_bw = 1.0/(0.00001 + abs(sort_bw))
                sort_cw = 1.0/(0.00001 + abs(sort_cw))
                sort_dw = 1.0/(0.00001 + abs(sort_dw))
                
                ab_frac = (sort_bw) / (sort_aw + sort_bw)
                ba_frac = 1.0-ab_frac
                #ba_frac = (sort_aw) / (sort_aw + sort_bw)
                ac_frac = (sort_cw) / (sort_aw + sort_cw)
                ca_frac = 1.0-ac_frac
                #ca_frac = (sort_aw) / (sort_aw + sort_cw)
                ad_frac = (sort_dw) / (sort_aw + sort_dw)
                da_frac = 1.0-ad_frac
                #da_frac = (sort_aw) / (sort_aw + sort_dw)
           
                ab = ba_frac*(sort_a) + (ab_frac)*sort_b
                ad = da_frac*(sort_a) + (ad_frac)*sort_d
                ac = ca_frac*(sort_a) + (ac_frac)*sort_c
                self.Vertices_CPU += [ab]
                self.Vertices_CPU += [ad]
                self.Vertices_CPU += [ac]

        self.Vertices_CPU = np.array(self.Vertices_CPU)
        return self.Vertices_CPU


    def MT(self):
        iso = 0
        convVal = 32767.0
        for voxid in range(0,self.Size[2]):
        #for voxid in range(0,36):    
            idx = voxid *4
            ax = 0.0  
            ay = 0.0  
            az = 0.0
            bx = 0.0  
            by = 0.0  
            bz = 0.0
            cx = 0.0  
            cy = 0.0  
            cz = 0.0

            ax = self.VertexArray[(self.VoxelArray[idx+0])*3+0]  
            ay = self.VertexArray[(self.VoxelArray[idx+0])*3+1] 
            az = self.VertexArray[(self.VoxelArray[idx+0])*3+2] 

            bx = self.VertexArray[(self.VoxelArray[idx+1])*3+0]  
            by = self.VertexArray[(self.VoxelArray[idx+1])*3+1] 
            bz = self.VertexArray[(self.VoxelArray[idx+1])*3+2] 

            cx = self.VertexArray[(self.VoxelArray[idx+2])*3+0]  
            cy = self.VertexArray[(self.VoxelArray[idx+2])*3+1] 
            cz = self.VertexArray[(self.VoxelArray[idx+2])*3+2] 

            dx = self.VertexArray[(self.VoxelArray[idx+3])*3+0] 
            dy = self.VertexArray[(self.VoxelArray[idx+3])*3+1] 
            dz = self.VertexArray[(self.VoxelArray[idx+3])*3+2] 
             
            aw = float(self.TSDFTable[idx+0])/ convVal
            bw = float(self.TSDFTable[idx+1])/ convVal
            cw = float(self.TSDFTable[idx+2])/ convVal
            dw = float(self.TSDFTable[idx+3])/ convVal

            self.aw[idx] = aw 
            self.bw[idx] = bw 
            self.cw[idx] = cw 
            self.dw[idx] = dw 

             
            count =0
            if(aw >= iso):
                count +=1
            if(bw >= iso):
                count +=1
            if(cw >=iso):
                count +=1
            if(dw >= iso):
                count +=1
            #aw = round(aw)
            #bw = round(bw)
            #cw = round(cw)
            #dw = round(dw)
            if (count==0 or count==4):
                self.Vertices[18*(voxid)+0] = 0.0
                self.Vertices[18*(voxid)+1] = 0.0          
                self.Vertices[18*(voxid)+2] = 0.0

                self.Vertices[18*(voxid)+3] = 0.0
                self.Vertices[18*(voxid)+4] = 0.0
                self.Vertices[18*(voxid)+5] = 0.0

                self.Vertices[18*(voxid)+6] = 0.0
                self.Vertices[18*(voxid)+7] = 0.0
                self.Vertices[18*(voxid)+8] = 0.0

                self.Vertices[18*(voxid)+9] = 0.0
                self.Vertices[18*(voxid)+10] = 0.0
                self.Vertices[18*(voxid)+11] = 0.0
                
                self.Vertices[18*(voxid)+12] = 0.0
                self.Vertices[18*(voxid)+13] = 0.0
                self.Vertices[18*(voxid)+14] = 0.0
                
                self.Vertices[18*(voxid)+15] = 0.0
                self.Vertices[18*(voxid)+16] = 0.0
                self.Vertices[18*(voxid)+17] = 0.0
            elif (count == 3):

                if (dw < iso): 
             
                    sort_ax = ax
                    sort_ay = ay
                    sort_az = az
                    sort_aw = aw

                    sort_bx = bx
                    sort_by = by
                    sort_bz = bz
                    sort_bw = bw

                    sort_cx = cx
                    sort_cy = cy
                    sort_cz = cz
                    sort_cw = cw

                    sort_dx = dx
                    sort_dy = dy
                    sort_dz = dz
                    sort_dw = dw
                
                elif (cw < iso):
                
                    sort_ax = ax
                    sort_ay = ay
                    sort_az = az
                    sort_aw = aw 
                    sort_bx = dx
                    sort_by = dy
                    sort_bz = dz
                    sort_bw = dw
                    sort_cx = bx
                    sort_cy = by
                    sort_cz = bz
                    sort_cw = bw
                    sort_dx = cx
                    sort_dy = cy
                    sort_dz = cz
                    sort_dw = cw
                
                elif (bw  < iso): 

                    sort_ax = ax
                    sort_ay = ay
                    sort_az = az
                    sort_aw = aw

                    sort_bx = cx
                    sort_by = cy
                    sort_bz = cz
                    sort_bw = cw

                    sort_cx = dx
                    sort_cy = dy
                    sort_cz = dz
                    sort_cw = dw

                    sort_dx = bx
                    sort_dy = by
                    sort_dz = bz
                    sort_dw = bw
                
                else :
                    sort_ax = bx
                    sort_ay = by
                    sort_az = bz
                    sort_aw = bw

                    sort_bx = dx
                    sort_by = dy
                    sort_bz = dz
                    sort_bw = dw


                    sort_cx = cx
                    sort_cy = cy
                    sort_cz = cz
                    sort_cw = cw

                    sort_dx = ax
                    sort_dy = ay
                    sort_dz = az
                    sort_dw = aw
    
            
                sort_aw = 1.0/(0.00001 + abs(sort_aw))
                sort_bw = 1.0/(0.00001 + abs(sort_bw))
                sort_cw = 1.0/(0.00001 + abs(sort_cw))
                sort_dw = 1.0/(0.00001 + abs(sort_dw))

                #bu degil
                da_frac = (sort_aw) / (sort_aw + sort_dw)
                ad_frac = 1.0-da_frac
                #da_frac = (sort_dw) / (sort_aw + sort_dw)
                db_frac = (sort_bw) / (sort_bw + sort_dw)
                bd_frac = 1.0-db_frac
                #db_frac = (sort_dw) / (sort_bw + sort_dw)
                dc_frac = (sort_cw) / (sort_cw + sort_dw)
                cd_frac = 1.0-dc_frac
                #dc_frac = (sort_dw) / (sort_cw + sort_dw)
                
   
                adx = da_frac*(sort_ax) + (ad_frac)*sort_dx
                ady = da_frac*(sort_ay) + (ad_frac)*sort_dy
                adz = da_frac*(sort_az) + (ad_frac)*sort_dz

                bdx = db_frac*(sort_bx) + (bd_frac)*sort_dx
                bdy = db_frac*(sort_by) + (bd_frac)*sort_dy
                bdz = db_frac*(sort_bz) + (bd_frac)*sort_dz
                
                cdx = dc_frac*(sort_cx) + (cd_frac)*sort_dx
                cdy = dc_frac*(sort_cy) + (cd_frac)*sort_dy
                cdz = dc_frac*(sort_cz) + (cd_frac)*sort_dz


           
                self.Vertices[18*(voxid)] = adx
                self.Vertices[18*(voxid)+1] = ady
                self.Vertices[18*(voxid)+2] = adz
                
                self.Vertices[18*(voxid)+3] =bdx
                self.Vertices[18*(voxid)+4] =bdy
                self.Vertices[18*(voxid)+5] =bdz
                
                self.Vertices[18*(voxid)+6] = cdx
                self.Vertices[18*(voxid)+7] = cdy
                self.Vertices[18*(voxid)+8] = cdz
                
                
                self.Vertices[18*(voxid)+9] = 0.0
                self.Vertices[18*(voxid)+10] = 0.0
                self.Vertices[18*(voxid)+11] = 0.0
                
                self.Vertices[18*(voxid)+12] = 0.0
                self.Vertices[18*(voxid)+13] = 0.0
                self.Vertices[18*(voxid)+14] = 0.0
                
                self.Vertices[18*(voxid)+15] = 0.0
                self.Vertices[18*(voxid)+16] = 0.0
                self.Vertices[18*(voxid)+17] = 0.0
                

            elif (count == 2) :

                if (aw >= iso and bw >= iso) :
                
                    sort_ax = ax
                    sort_ay = ay
                    sort_az = az
                    sort_aw = aw

                    sort_bx = bx
                    sort_by = by
                    sort_bz = bz
                    sort_bw = bw

                    sort_cx = cx
                    sort_cy = cy
                    sort_cz = cz
                    sort_cw = cw

                    sort_dx = dx
                    sort_dy = dy
                    sort_dz = dz
                    sort_dw = dw
                
                elif (aw >= iso and cw >= iso) :
                    
                    sort_ax = ax
                    sort_ay = ay
                    sort_az = az
                    sort_aw = aw

                    sort_bx = cx
                    sort_by = cy
                    sort_bz = cz
                    sort_bw = cw

                    sort_cx = dx
                    sort_cy = dy
                    sort_cz = dz
                    sort_cw = dw

                    sort_dx = bx
                    sort_dy = by
                    sort_dz = bz
                    sort_dw = bw
                elif (aw >= iso and dw >= iso) :          
                
                    sort_ax = ax
                    sort_ay = ay
                    sort_az = az
                    sort_aw = aw 
                    sort_bx = dx
                    sort_by = dy
                    sort_bz = dz
                    sort_bw = dw
                    sort_cx = bx
                    sort_cy = by
                    sort_cz = bz
                    sort_cw = bw
                    sort_dx = cx
                    sort_dy = cy
                    sort_dz = cz
                    sort_dw = cw
                
                elif (bw >= iso and cw >= iso) :
                
                    sort_ax = bx
                    sort_ay = by
                    sort_az = bz
                    sort_aw = bw 
                  
                    sort_bx = cx
                    sort_by = cy
                    sort_bz = cz
                    sort_bw = cw
                 
                    sort_cx = ax
                    sort_cy = ay
                    sort_cz = az
                    sort_cw = aw
                  
                    sort_dx = dx
                    sort_dy = dy
                    sort_dz = dz
                    sort_dw = dw
                
                elif (bw >= iso and dw >= iso) :
                
                    sort_ax = bx
                    sort_ay = by
                    sort_az = bz
                    sort_aw = bw

                    sort_bx = dx
                    sort_by = dy
                    sort_bz = dz
                    sort_bw = dw


                    sort_cx = cx
                    sort_cy = cy
                    sort_cz = cz
                    sort_cw = cw

                    sort_dx = ax
                    sort_dy = ay
                    sort_dz = az
                    sort_dw = aw
                
                else :
                
                    sort_ax = cx
                    sort_ay = cy
                    sort_az = cz
                    sort_aw = cw
              
                    sort_bx = dx
                    sort_by = dy
                    sort_bz = dz
                    sort_bw = dw

                    sort_cx = ax
                    sort_cy = ay
                    sort_cz = az
                    sort_cw = aw
              
                    sort_dx = bx
                    sort_dy = by
                    sort_dz = bz
                    sort_dw = bw
            
                
                sort_aw = 1.0/(0.00001 + abs(sort_aw))
                sort_bw = 1.0/(0.00001 + abs(sort_bw))
                sort_cw = 1.0/(0.00001 + abs(sort_cw))
                sort_dw = 1.0/(0.00001 + abs(sort_dw))
                
                ac_frac = (sort_cw) / (sort_aw + sort_cw)
                ca_frac = 1.0-ac_frac
                #ca_frac = (sort_aw) / (sort_aw + sort_cw)
                ad_frac = (sort_dw) / (sort_aw + sort_dw)
                da_frac = 1.0-ad_frac
                #da_frac = (sort_aw) / (sort_aw + sort_dw)
                bc_frac = (sort_cw) / (sort_bw + sort_cw)
                cb_frac = 1.0-bc_frac
                #cb_frac = (sort_bw) / (sort_bw + sort_cw)
                bd_frac = (sort_dw) / (sort_bw + sort_dw)
                db_frac = 1.0-bd_frac
                #db_frac = (sort_bw) / (sort_bw + sort_dw)
                '''
                ac_frac = 0.5
                ca_frac = 0.5
                ad_frac = 0.5
                da_frac = 0.5
                bc_frac = 0.5
                cb_frac = 0.5
                bd_frac = 0.5
                db_frac = 0.5
                '''
                adx = da_frac*(sort_ax) + (ad_frac)*sort_dx
                ady = da_frac*(sort_ay) + (ad_frac)*sort_dy
                adz = da_frac*(sort_az) + (ad_frac)*sort_dz
                bdx = db_frac*(sort_bx) + (bd_frac)*sort_dx
                bdy = db_frac*(sort_by) + (bd_frac)*sort_dy
                bdz = db_frac*(sort_bz) + (bd_frac)*sort_dz
                acx = ca_frac*(sort_ax) + (ac_frac)*sort_cx
                acy = ca_frac*(sort_ay) + (ac_frac)*sort_cy
                acz = ca_frac*(sort_az) + (ac_frac)*sort_cz
                bcx = cb_frac*(sort_bx) + (bc_frac)*sort_cx
                bcy = cb_frac*(sort_by) + (bc_frac)*sort_cy
                bcz = cb_frac*(sort_bz) + (bc_frac)*sort_cz
            
            
                self.Vertices[18*(voxid)] = acx;
                self.Vertices[18*(voxid)+1] = acy;
                self.Vertices[18*(voxid)+2] = acz;
                
                self.Vertices[18*(voxid)+3] = bcx;
                self.Vertices[18*(voxid)+4] =bcy;
                self.Vertices[18*(voxid)+5] = bcz;
                
                self.Vertices[18*(voxid)+6] = adx;
                self.Vertices[18*(voxid)+7] = ady;
                self.Vertices[18*(voxid)+8] = adz;

              
                self.Vertices[18*(voxid)+9] = bcx;
                self.Vertices[18*(voxid)+10] = bcy;
                self.Vertices[18*(voxid)+11] = bcz;
                
                self.Vertices[18*(voxid)+12] = bdx;
                self.Vertices[18*(voxid)+13] = bdy;
                self.Vertices[18*(voxid)+14] = bdz;
                
                self.Vertices[18*(voxid)+15] = adx;
                self.Vertices[18*(voxid)+16] =ady;
                self.Vertices[18*(voxid)+17] = adz;
                
            elif (count == 1) :
                '''    
                self.Vertices[18*(voxid)+0] = 0.0
                self.Vertices[18*(voxid)+1] = 0.0          
                self.Vertices[18*(voxid)+2] = 0.0

                self.Vertices[18*(voxid)+3] = 0.0
                self.Vertices[18*(voxid)+4] = 0.0
                self.Vertices[18*(voxid)+5] = 0.0

                self.Vertices[18*(voxid)+6] = 0.0
                self.Vertices[18*(voxid)+7] = 0.0
                self.Vertices[18*(voxid)+8] = 0.0

                self.Vertices[18*(voxid)+9] = 0.0
                self.Vertices[18*(voxid)+10] = 0.0
                self.Vertices[18*(voxid)+11] = 0.0
                
                self.Vertices[18*(voxid)+12] = 0.0
                self.Vertices[18*(voxid)+13] = 0.0
                self.Vertices[18*(voxid)+14] = 0.0
                
                self.Vertices[18*(voxid)+15] = 0.0
                self.Vertices[18*(voxid)+16] = 0.0
                self.Vertices[18*(voxid)+17] = 0.0
                '''
                if (aw >= iso):
                
                    sort_ax = ax
                    sort_ay = ay
                    sort_az = az
                    sort_aw = aw

                    sort_bx = bx
                    sort_by = by
                    sort_bz = bz
                    sort_bw = bw

                    sort_cx = cx
                    sort_cy = cy
                    sort_cz = cz
                    sort_cw = cw

                    sort_dx = dx
                    sort_dy = dy
                    sort_dz = dz
                    sort_dw = dw
                
                elif (bw >= iso):
                    sort_ax = bx
                    sort_ay = by
                    sort_az = bz
                    sort_aw = bw 
                  
                    sort_bx = cx
                    sort_by = cy
                    sort_bz = cz
                    sort_bw = cw
                 
                    sort_cx = ax
                    sort_cy = ay
                    sort_cz = az
                    sort_cw = aw
                  
                    sort_dx = dx
                    sort_dy = dy
                    sort_dz = dz
                    sort_dw = dw 

                
                elif (cw >= iso) :
                
                  
                    sort_ax = cx
                    sort_ay = cy
                    sort_az = cz
                    sort_aw = cw
                    sort_bx = ax
                    sort_by = ay
                    sort_bz = az
                    sort_bw = aw
                    sort_cx = bx
                    sort_cy = by
                    sort_cz = bz
                    sort_cw = bw
                    sort_dx = dx
                    sort_dy = dy
                    sort_dz = dz
                    sort_dw = dw
                
                else :
                
                    sort_ax = dx
                    sort_ay = dy
                    sort_az = dz
                    sort_aw = dw
                    sort_bx = cx
                    sort_by = cy
                    sort_bz = cz
                    sort_bw = cw
                 
                    sort_cx = bx
                    sort_cy = by
                    sort_cz = bz
                    sort_cw = bw
                    sort_dx = ax
                    sort_dy = ay
                    sort_dz = az
                    sort_dw = aw 
                
                
                sort_aw = 1.0/(0.00001 + abs(sort_aw))
                sort_bw = 1.0/(0.00001 + abs(sort_bw))
                sort_cw = 1.0/(0.00001 + abs(sort_cw))
                sort_dw = 1.0/(0.00001 + abs(sort_dw))
                
                ab_frac = (sort_bw) / (sort_aw + sort_bw)
                ba_frac = 1.0-ab_frac
                #ba_frac = (sort_aw) / (sort_aw + sort_bw)
                ac_frac = (sort_cw) / (sort_aw + sort_cw)
                ca_frac = 1.0-ac_frac
                #ca_frac = (sort_aw) / (sort_aw + sort_cw)
                ad_frac = (sort_dw) / (sort_aw + sort_dw)
                da_frac = 1.0-ad_frac
                #da_frac = (sort_aw) / (sort_aw + sort_dw)
           
                
                
                abx = ba_frac*(sort_ax) + (ab_frac)*sort_bx
                aby = ba_frac*(sort_ay) + (ab_frac)*sort_by
                abz = ba_frac*(sort_az) + (ab_frac)*sort_bz

                adx = da_frac*(sort_ax) + (ad_frac)*sort_dx
                ady = da_frac*(sort_ay) + (ad_frac)*sort_dy
                adz = da_frac*(sort_az) + (ad_frac)*sort_dz

                acx = ca_frac*(sort_ax) + (ac_frac)*sort_cx
                acy = ca_frac*(sort_ay) + (ac_frac)*sort_cy
                acz = ca_frac*(sort_az) + (ac_frac)*sort_cz
          
                self.Vertices[18*(voxid)] = abx
                self.Vertices[18*(voxid)+1] = aby
                self.Vertices[18*(voxid)+2] = abz
                
                self.Vertices[18*(voxid)+3] = adx
                self.Vertices[18*(voxid)+4] = ady
                self.Vertices[18*(voxid)+5] = adz
                
                self.Vertices[18*(voxid)+6] = acx
                self.Vertices[18*(voxid)+7] = acy
                self.Vertices[18*(voxid)+8] = acz
                self.Vertices[18*(voxid)+9] = 0.0
                self.Vertices[18*(voxid)+10] = 0.0
                self.Vertices[18*(voxid)+11] = 0.0
            
                self.Vertices[18*(voxid)+12] = 0.0
                self.Vertices[18*(voxid)+13] = 0.0
                self.Vertices[18*(voxid)+14] = 0.0
                
                self.Vertices[18*(voxid)+15] = 0.0
                self.Vertices[18*(voxid)+16] = 0.0
                self.Vertices[18*(voxid)+17] = 0.0
                
    def SaveToPly_CPU(self, savepath, BGR=None):
        with open(savepath, "w") as fs:
            # Write headers
            fs.write("ply\n")
            fs.write("format ascii 1.0\n")
            fs.write("comment ply file \n")
            fs.write("element vertex %d \n" %(len(self.Vertices_CPU)))
            fs.write("property float x\n")
            fs.write("property float y\n")
            fs.write("property float z\n")
            if BGR is not None:
                fs.write("property uchar red\n")
                fs.write("property uchar green\n")
                fs.write("property uchar blue\n")
            fs.write("element face %d \n" %(int(len(self.Vertices_CPU)/3)))
            fs.write("property list uchar int vertex_indices\n")
            fs.write("end_header\n")
            
            if BGR is not None:
                for v, bgr in zip(self.Vertices_CPU, BGR):
                    fs.write("%f %f %f %d %d %d\n" %(v[0], v[1], v[2], int(bgr[2]), int(bgr[1]), int(bgr[0])))
            else:
                for v in self.Vertices_CPU:
                    fs.write("%f %f %f\n" %(v[0], v[1], v[2]))
            # Faces
            for i in range(int(len(self.Vertices_CPU)/3)):
                fs.write("3 %d %d %d \n" %(i*3, (i*3)+1, (i*3)+2)) 
            

    def SaveToPly(self, name, display = 0):
        """
        Function to record the created mesh into a .ply file
        Create file .ply with vertices and faces for vizualization in MeshLab
        :param name: string, name of the file
        :param times: 0 if you do not want to display the time
        :return: none
        """
        #Count how many faces are there really

        PLYVertices = np.zeros((9*self.Size[2], 3), dtype = np.float32) #bunlari ravel 
        PLYVertices = PLYVertices.ravel() #672*9
        if display != 0:
            start_time3 = time.time()
        #path = '../meshes/'

            
        V = self.Vertices.reshape(int(len(self.Vertices)/3), 3)
        valid = np.any(V>0, axis=1)
        valid2 = np.any(valid.reshape(int(len(valid)/3), 3), axis=1)
        valid2 = np.vstack([valid2, valid2, valid2]).T.flatten()
        V = V[valid2]
        with open(name, 'w') as f:
            print(V.shape)
            # Write headers
            f.write("ply\n")
            f.write("format ascii 1.0\n")
            f.write("comment ply file \n")
            f.write("element vertex %d \n" %(len(V)))
            f.write("property float x\n")
            f.write("property float y\n")
            f.write("property float z\n")
            f.write("element face %d \n" %(int(len(V)/3)))
            f.write("property list uchar int vertex_indices\n")
            f.write("end_header\n")
            # Vertices
            for v in V:
                f.write("%f %f %f\n" %(v[0], v[1], v[2]))
            # Faces
            for i in range(int(len(V)/3)):
                if(i > 19947):
                    print "3 %d %d %d" %(i*3, (i*3)+1, (i*3)+2)
                f.write("3 %d %d %d \n" %(i*3, (i*3)+1, (i*3)+2)) 



        # # Write vertices\
        # count = 0 
        # for cc in range(0,self.Size[2]*2): #5184
        #     c = cc * 9 
        #     if not ((self.Vertices[c] == 0.0) and (self.Vertices[c+1] == 0.0) and (self.Vertices[c+2] == 0.0) \
        #         and (self.Vertices[c+3] == 0.0) and (self.Vertices[c+4] == 0.0) and (self.Vertices[c+5] == 0.0) \
        #         and (self.Vertices[c+6] == 0.0) and (self.Vertices[c+7] == 0.0) and (self.Vertices[c+8] == 0.0) ):
        #             f.write("%f %f %f \n" %((self.Vertices[c]), (self.Vertices[c+1]), (self.Vertices[c+2])))
        #             f.write("%f %f %f \n" %((self.Vertices[c+3]), (self.Vertices[c+4]), (self.Vertices[c+5])))
        #             f.write("%f %f %f \n" %((self.Vertices[c+6]), (self.Vertices[c+7]), (self.Vertices[c+8])))
        #             count +=3
        # # Write the faces
        # for i in range(0,self.count): #624
        #     f.write("3 %d %d %d \n" %(i*3, (i*3)+1, (i*3)+2)) 
        # print("count in saveply")
        # print(count)             
        # f.close()

        if display != 0:
            elapsed_time = time.time() - start_time3
            print("SaveToPly: {0:f}".format(elapsed_time))

    def SaveToPlyRunTwoPass(self, name, display = 0):
        """
        Function to record the created mesh into a .ply file
        Create file .ply with vertices and faces for vizualization in MeshLab
        :param name: string, name of the file
        :param times: 0 if you do not want to display the time
        :return: none
        """
        #Count how many faces are there really

        PLYVertices = np.zeros((9*self.Size[2], 3), dtype = np.float32) #bunlari ravel 
        PLYVertices = PLYVertices.ravel() #672*9
        if display != 0:
            start_time3 = time.time()
        #path = '../meshes/'
        f = open(name, 'wb')
        
        # Write headers
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write("comment ply file \n")
        f.write("element vertex %d \n" %(self.count*3))
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        #f.write("property float nx\n")
        #f.write("property float ny\n")
        #f.write("property float nz\n")
        f.write("element face %d \n" %(self.count))
        f.write("property list uchar int vertex_indices\n")
        f.write("end_header\n")
         
        # Write vertices\
        
        for cc in range(0,self.count): #5184
            c = cc * 9
            n = cc * 3
            #f.write("%f %f %f %f %f %f\n" %((self.Vertices[c]), (self.Vertices[c+1]), (self.Vertices[c+2]),(self.Normales[n]),(self.Normales[n+1]),(self.Normales[n+2])))
            #f.write("%f %f %f %f %f %f\n" %((self.Vertices[c+3]), (self.Vertices[c+4]), (self.Vertices[c+5]),(self.Normales[n]),(self.Normales[n+1]),(self.Normales[n+2])))
            #f.write("%f %f %f %f %f %f\n" %((self.Vertices[c+6]), (self.Vertices[c+7]), (self.Vertices[c+8]),(self.Normales[n]),(self.Normales[n+1]),(self.Normales[n+2])))
            f.write("%f %f %f\n" %((self.Verticestwopass[c]), (self.Verticestwopass[c+1]), (self.Verticestwopass[c+2])))
            f.write("%f %f %f\n" %((self.Verticestwopass[c+3]), (self.Verticestwopass[c+4]), (self.Verticestwopass[c+5])))
            f.write("%f %f %f\n" %((self.Verticestwopass[c+6]), (self.Verticestwopass[c+7]), (self.Verticestwopass[c+8])))

        # Write the faces
        for i in range(0,self.count): #624
            f.write("3 %d %d %d \n" %(i*3, (i*3)+1, (i*3)+2))             
        f.close()

        if display != 0:
            elapsed_time = time.time() - start_time3
            print("SaveToPly: {0:f}".format(elapsed_time))