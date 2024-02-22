import numpy as np
import pyCT.forward as forward

def getDataloader(mode, header, config, *args):
    if mode == 'train':
        return _DataloaderTraining(header, config, *args)
    elif mode == 'infer':
        return _DataloaderInference(header, config)
    
    
class _DataloaderInference():
    
    def __init__(self, header, config):
        self.__header = header
        self.__config = config
        self.__loadInput()
        self.__setBatch()
        
    def __loadInput(self):
        sx, sy, sz = self.__header.object.length.x, self.__header.object.length.y, self.__header.object.length.z
        nx, ny, nz = self.__header.object.size.x, self.__header.object.size.y, self.__header.object.size.z
        dx, dy, dz = self.__header.object.spacing.x, self.__header.object.spacing.y, self.__header.object.spacing.z
        near,far = self.__header.distance.near, self.__header.distance.far

        if nx == 1:
            x = [0]
        else:
            x = np.linspace(-(sx-dx)/2, (sx-dx)/2, nx)
        if ny == 1:
            y = [0]
        else:
            y = np.linspace(-(sy-dy)/2, (sy-dy)/2, ny)
        if nz == 1:
            z = [0]
        else:
            z = np.linspace(-(sz-dz)/2, (sz-dz)/2, nz)
        x,y,z = np.meshgrid(x,y,z,indexing='ij')
        
        self.input = np.stack([x,y,z], axis=-1).reshape(-1,3) / (far-near)
        self.num_input = nx*ny*nz

    def __setBatch(self):
        num_input = self.num_input
        batch_size = self.__config.batch.size
        num_batch = num_input // batch_size
        if num_input % batch_size > 0:
            num_batch += 1
        self.num_batch = num_batch

    def generateBatch(self):
        batch_size = self.__config.batch.size
        for batch_idx in range(self.num_batch):
            yield self.input[batch_idx*batch_size : (batch_idx+1)*batch_size]
    
    
    
class _DataloaderTraining():
    
    def __init__(self, header, config, *args):
        self.__header = header
        self.__config = config
        self.__data = args[0]
        self.__angles = args[1]

        self.origins = None
        self.directions = None
        self.data = None
        self.num_data = None
        self.num_batch = None
        self.__loadAllData()
        self.__setBatch()

    def __getCameraTransformation(self, angle):
        frame = np.array([[np.cos(angle), -np.sin(angle), 0], [np.sin(angle), np.cos(angle), 0], [0, 0, 1]]) @ np.array([[0,0,1],[1,0,0],[0,1,0]])
        frame = frame.T
        origin = self.__header.distance.source2object * frame[2]
        translation = np.eye(4)
        translation[:-1, -1] = -origin
        rotation = np.eye(4)
        rotation[:-1, :-1] = frame
        
        return rotation@translation
    
    def __castRays(self, angle):
        cameraTransformation = self.__getCameraTransformation(angle)
        cameraTransformation = np.linalg.inv(cameraTransformation)
        
        su, sv = self.__header.detector.length.u, self.__header.detector.length.v
        nu, nv = self.__header.detector.size.u, self.__header.detector.size.v
        du, dv = self.__header.detector.spacing.u, self.__header.detector.spacing.v

        X, Y = np.meshgrid(du*np.arange(nu)-su/2+du/2, dv*np.arange(nv)-sv/2+dv/2)
        if not self.__header.mode:
            Z = np.zeros_like(X)
            ray_origin = np.stack([X,Y,Z], axis=-1) @ cameraTransformation[:-1,:-1].T + cameraTransformation[:-1,-1]
            ray_direction = np.broadcast_to(-cameraTransformation[:-1,2], ray_origin.shape)
        else:
            Z = -self.__header.distance.source2detector * np.ones_like(X)
            ray_direction = np.stack([X,Y,Z], axis=-1) @ cameraTransformation[:-1,:-1].T
            ray_direction /= np.linalg.norm(ray_direction, axis=-1, keepdims=True)
            ray_origin = np.broadcast_to(cameraTransformation[:-1,-1], ray_direction.shape)
        
        return ray_origin, ray_direction
    
        
    def __loadAllData(self):
        origins = []
        directions = []
        
        self.data = self.__data.flatten()
        self.num_data = self.data.size
        for angle in self.__angles:
            ray_origin, ray_direction = self.__castRays(angle)
            origins.append(ray_origin.reshape(-1,3))
            directions.append(ray_direction.reshape(-1,3))
        self.origins = np.concatenate(origins)
        self.directions = np.concatenate(directions)
        

    def __samplePointsOnRay(self, ray_origin, ray_direction):
        o = ray_origin[...,None,:]
        d = ray_direction[...,None,:]
        
        near = self.__header.distance.near
        far = self.__header.distance.far
        step = self.__config.ray_sampling.step
        sampling_num = int((far-near)//step)
        t = np.linspace(0, 1, sampling_num+1)[:-1]
        if self.__config.ray_sampling.mode == 'sample':
            t = near*(1-t) + far*t
            t = np.tile(t, len(ray_origin)).reshape(-1, sampling_num)
            t = t[..., None]
        elif self.__config.ray_sampling.mode == 'stratified':
            mids = (t[1:]+t[:-1]) / 2
            uppers = np.concatenate([mids,[1]])
            lowers = np.concatenate([[0],mids])
            t = lowers + (uppers-lowers) * np.random.rand(len(o), sampling_num)
            t = t[..., None]
            t = near*(1-t) + far*t
        dt = np.concatenate([np.diff(t[...,0]), far-t[:,-1]], axis=-1)
        return o + t*d, dt
    

    def __setBatch(self, batch_last=True):
        batch_size = self.__config.batch.size
        self.num_batch = self.num_data // batch_size
        if batch_last:
            if self.num_data % batch_size > 0:
                self.num_batch += 1


    def generateBatch(self, shuffle=False):
        batch_size = self.__config.batch.size
        if shuffle:
            all_idx = np.random.permutation(self.num_data)
        else:
            all_idx = np.arange(self.num_data)

        for batch_idx in range(self.num_batch):
            shuffle_idx = all_idx[batch_idx*batch_size : (batch_idx+1)*batch_size]
            o = self.origins[shuffle_idx]
            d = self.directions[shuffle_idx]
            y = self.data[shuffle_idx]
            x, dt = self.__samplePointsOnRay(o, d)
            yield x, dt, y