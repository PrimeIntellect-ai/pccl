from typing import Union, Optional, Tuple, Dict, List
import time
from ipaddress import ip_address, IPv4Address, IPv6Address
from enum import Enum
from pccl._loader import load_native_module
import logging
import importlib
import pccl._cuda


class ModuleDummy:
    def __init__(self, name: str):
        self.name = name

    def __getattr__(self, name: str):
        raise RuntimeError(
            f"Module {self.name} is not available. Please install the module via pip if you want pccl to interoperate with it.")


# Check if torch is available
if importlib.util.find_spec('torch') is not None:
    import torch
else:
    torch = ModuleDummy('torch')

# Check if numpy is available
if importlib.util.find_spec('numpy') is not None:
    import numpy as np
else:
    np = ModuleDummy('numpy')

# Enable faulthandler for debugging
PY_PCCL_DEBUG = False
if PY_PCCL_DEBUG:
    import faulthandler

    faulthandler.enable()

ffi, C = load_native_module()  # Load native module


class Result(Enum):
    """PCCL result codes."""
    SUCCESS = C.pcclSuccess
    NOT_INITIALIZED = C.pcclNotInitialized
    INTERNAL_ERROR = C.pcclInternalError
    INVALID_ARGUMENT = C.pcclInvalidArgument
    INVALID_USAGE = C.pcclInvalidUsage
    TOO_FEW_PEERS = C.pcclTooFewPeers
    MASTER_CONNECTION_FAILED = C.pcclMasterConnectionFailed
    RANK_CONNECTION_FAILED = C.pcclRankConnectionFailed
    RANK_CONNECTION_LOST = C.pcclRankConnectionLost
    NO_SHARED_STATE_AVAILABLE = C.pcclNoSharedStateAvailable
    PENDING_ASYNC_OPS = C.pcclPendingAsyncOps
    UPDATE_TOPOLOGY_FAILED = C.pcclUpdateTopologyFailed
    TOPOLOGY_OPTIMIZATION_FAILED = C.pcclTopologyOptimizationFailed


class PCCLError(Exception):
    """PCCL specific exception."""

    def __init__(self, result: Result, func_name: str):
        super().__init__(f'{func_name} failed with error: {result.name}')
        self.result = result

    def __str__(self):
        return f'{super().__str__()}: {self.result.name}'

    @staticmethod
    def check(result: int, func_name: str):
        """Check the result and raise an exception if necessary."""
        if result != Result.SUCCESS.value:
            raise PCCLError(Result(result), func_name)


# Init PCCL
PCCLError.check(C.pcclInit(), "pcclInit")


# Get PCCL build info

def __get_build_info() -> Dict[str, any]:
    build_info = ffi.new('pcclBuildInfo_t*')
    PCCLError.check(C.pcclGetBuildInfo(build_info), "pcclGetBuildInfo")
    return {
        'has_cuda_support': build_info.has_cuda_support
    }


_build_info = __get_build_info()
pccl._cuda.is_cuda_available = _build_info['has_cuda_support']


class ReduceOp(Enum):
    """PCCL reduction operations."""
    SUM = C.pcclSum
    AVG = C.pcclAvg
    PROD = C.pcclProd
    MAX = C.pcclMax
    MIN = C.pcclMin


class Attribute(Enum):
    """PCCL attributes."""
    GLOBAL_WORLD_SIZE = C.PCCL_ATTRIBUTE_GLOBAL_WORLD_SIZE
    LOCAL_WORLD_SIZE = C.PCCL_ATTRIBUTE_PEER_GROUP_WORLD_SIZE
    NUM_DISTINCT_PEER_GROUPS = C.PCCL_ATTRIBUTE_NUM_DISTINCT_PEER_GROUPS
    LARGEST_PEER_GROUP_WORLD_SIZE = C.PCCL_ATTRIBUTE_LARGEST_PEER_GROUP_WORLD_SIZE

class DataType(Enum):
    """PCCL primitive data types."""
    UINT8 = C.pcclUint8
    INT8 = C.pcclInt8
    UINT16 = C.pcclUint16
    UINT32 = C.pcclUint32
    INT32 = C.pcclInt32
    UINT64 = C.pcclUint64
    INT64 = C.pcclInt64
    FLOAT = C.pcclFloat
    DOUBLE = C.pcclDouble

    def to_torch_dtype(self):
        """Converts a DataType to the corresponding PyTorch dtype."""
        dtype_map = {
            DataType.UINT8: torch.uint8,
            DataType.INT8: torch.int8,
            DataType.UINT16: torch.uint16,
            DataType.UINT32: torch.uint32,
            DataType.INT32: torch.int32,
            DataType.UINT64: torch.uint64,
            DataType.INT64: torch.int64,
            DataType.FLOAT: torch.float32,
            DataType.DOUBLE: torch.float64,
        }
        assert self in dtype_map, f'Unsupported DataType: {self}'
        return dtype_map[self]

    def to_numpy_dtype(self):
        """Converts a DataType to the corresponding Numpy dtype."""
        dtype_map = {
            DataType.UINT8: np.dtypes.UInt8DType,
            DataType.INT8: np.dtypes.Int8DType,
            DataType.UINT16: np.dtypes.UInt16DType,
            DataType.UINT32: np.dtypes.UInt32DType,
            DataType.INT32: np.dtypes.Int32DType,
            DataType.UINT64: np.dtypes.UInt64DType,
            DataType.INT64: np.dtypes.Int64DType,
            DataType.FLOAT: np.dtypes.Float32DType,
            DataType.DOUBLE: np.dtypes.Float64DType,
        }
        dtype_clazz = type(self)
        assert dtype_clazz in dtype_map, f'Unsupported dtype: {dtype_clazz}'
        # noinspection PyTypeChecker
        return dtype_map[dtype_clazz]

    @classmethod
    def from_torch_dtype(cls, dtype: 'torch.dtype'):
        """Converts a PyTorch dtype to the corresponding DataType."""
        dtype_map = {
            torch.uint8: cls.UINT8,
            torch.int8: cls.INT8,
            torch.uint16: cls.UINT16,
            torch.uint32: cls.UINT32,
            torch.int32: cls.INT32,
            torch.uint64: cls.UINT64,
            torch.int64: cls.INT64,
            torch.float32: cls.FLOAT,
            torch.float64: cls.DOUBLE,
        }
        return dtype_map.get(dtype, None)

    @classmethod
    def from_numpy_dtype(cls, dtype: 'np.dtype'):
        """Converts a Numpy dtype to the corresponding DataType."""
        dtype_map = {
            np.dtypes.UInt8DType: DataType.UINT8,
            np.dtypes.Int8DType: DataType.INT8,
            np.dtypes.UInt16DType: DataType.UINT16,
            np.dtypes.UInt32DType: DataType.UINT32,
            np.dtypes.Int32DType: DataType.INT32,
            np.dtypes.UInt64DType: DataType.UINT64,
            np.dtypes.Int64DType: DataType.INT64,
            np.dtypes.Float32DType: DataType.FLOAT,
            np.dtypes.Float64DType: DataType.DOUBLE,
        }
        dtype_clazz = type(dtype)
        assert dtype_clazz in dtype_map, f'Unsupported dtype: {dtype}'
        # noinspection PyTypeChecker
        return dtype_map[dtype_clazz]


class DeviceType(Enum):
    CPU = C.pcclDeviceCpu
    CUDA = C.pcclDeviceCuda

    @classmethod
    def from_torch_device_type(cls, device_type: str):
        """Converts a PyTorch device type to the corresponding DeviceType."""
        device_map = {
            'cpu': cls.CPU,
            'cuda': cls.CUDA,
        }
        return device_map.get(device_type, None)


class DistributionHint(Enum):
    """PCCL distribution hints."""
    NONE = C.PCCL_DISTRIBUTION_HINT_NONE
    NORMAL = C.PCCL_NORMAL_DISTRIBUTION_HINT
    UNIFORM = C.PCCL_UNIFORM_DISTRIBUTION_HINT


class QuantizationAlgorithm(Enum):
    """PCCL quantization algorithms."""
    NONE = C.pcclQuantNone
    MIN_MAX = C.pcclQuantMinMax


class ReduceOperandDescriptor:
    def __init__(self, datatype: DataType, distribution_hint: DistributionHint = DistributionHint.NONE):
        self.datatype = datatype
        self.distribution_hint = distribution_hint

    def to_c(self):
        c_desc = ffi.new("pcclReduceOperandDescriptor_t*")
        c_desc.datatype = self.datatype.value
        c_desc.distribution_hint = self.distribution_hint.value
        return c_desc


class QuantizationOptions:
    def __init__(self, quantized_datatype: DataType = DataType.UINT8,
                 algorithm: QuantizationAlgorithm = QuantizationAlgorithm.MIN_MAX):
        self.quantized_datatype = quantized_datatype
        self.algorithm = algorithm

    def to_c(self):
        c_opts = ffi.new("pcclQuantizationOptions_t*")
        c_opts.quantized_datatype = self.quantized_datatype.value
        c_opts.algorithm = self.algorithm.value
        return c_opts


class ReduceDescriptor:
    def __init__(self, count: int, op: ReduceOp, tag: int,
                 operand_descriptor: ReduceOperandDescriptor,
                 quantization_options: QuantizationOptions):
        self.count = count
        self.op = op
        self.tag = tag
        self.operand_descriptor = operand_descriptor
        self.quantization_options = quantization_options

    def to_c(self):
        c_desc = ffi.new("pcclReduceDescriptor_t*")
        c_desc.count = self.count
        c_desc.op = self.op.value
        c_desc.tag = self.tag
        c_desc.src_descriptor.datatype = self.operand_descriptor.datatype.value
        c_desc.src_descriptor.distribution_hint = self.operand_descriptor.distribution_hint.value
        c_desc.quantization_options.quantized_datatype = self.quantization_options.quantized_datatype.value
        c_desc.quantization_options.algorithm = self.quantization_options.algorithm.value
        return c_desc


class ReduceOpDescriptor:

    def __init__(self, sendbuf_ptr, recvbuf_ptr, reduce_descriptor: ReduceDescriptor):
        self.sendbuf_ptr = sendbuf_ptr
        self.recvbuf_ptr = recvbuf_ptr
        self.reduce_descriptor = reduce_descriptor

    @staticmethod
    def from_torch(send: 'torch.Tensor', recv: 'torch.Tensor', reduce_descriptor: ReduceDescriptor):
        assert send.is_contiguous(), 'Input tensor must be contiguous'
        assert recv.is_contiguous(), 'Output tensor must be contiguous'
        assert send.device == recv.device, 'Input and output tensors must be on the same device'
        assert send.dtype == recv.dtype, 'Input and output tensors must have the same dtype'
        assert send.device.type == 'cpu', 'Only CPU tensors are supported'
        assert send.numel() == recv.numel(), 'Input and output tensors must have the same number of elements'
        return ReduceOpDescriptor(
            send.data_ptr(),
            recv.data_ptr(),
            reduce_descriptor
        )

    @staticmethod
    def from_numpy(send: 'np.ndarray', recv: 'np.ndarray', reduce_descriptor: ReduceDescriptor):
        assert send.flags['C_CONTIGUOUS'], 'Input tensor must be contiguous'
        assert recv.flags['C_CONTIGUOUS'], 'Output tensor must be contiguous'
        assert send.dtype == recv.dtype, 'Input and output tensors must have the same dtype'
        assert send.size == recv.size, 'Input and output tensors must have the same number of elements'
        return ReduceOpDescriptor(
            send.ctypes.data,
            recv.ctypes.data,
            reduce_descriptor
        )

    def to_c_inpl(self, c_desc):
        c_desc.sendbuf = ffi.cast('void*', self.sendbuf_ptr)
        c_desc.recvbuf = ffi.cast('void*', self.recvbuf_ptr)
        c_desc.descriptor.count = self.reduce_descriptor.count
        c_desc.descriptor.op = self.reduce_descriptor.op.value
        c_desc.descriptor.tag = self.reduce_descriptor.tag
        c_desc.descriptor.src_descriptor.datatype = self.reduce_descriptor.operand_descriptor.datatype.value
        c_desc.descriptor.src_descriptor.distribution_hint = self.reduce_descriptor.operand_descriptor.distribution_hint.value
        c_desc.descriptor.quantization_options.quantized_datatype = self.reduce_descriptor.quantization_options.quantized_datatype.value
        c_desc.descriptor.quantization_options.algorithm = self.reduce_descriptor.quantization_options.algorithm.value

    def to_c(self):
        c_desc = ffi.new("pcclReduceOpDescriptor_t*")
        self.to_c_inpl(c_desc)
        return c_desc


# Define TensorInfo and SharedState Classes
class TensorInfo:
    def __init__(self, name: str, data_ptr: int, *, numel: int, dtype: DataType, device_type: DeviceType,
                 allow_content_inequality: bool):
        if data_ptr == 0:
            raise ValueError('Invalid data pointer: nullptr')
        self.name = name
        self.data_ptr = data_ptr
        self.numel = numel
        self.dtype = dtype
        self.device_type = device_type
        self.allow_content_inequality = allow_content_inequality

    @classmethod
    def from_torch(cls, tensor: 'torch.Tensor', name: str, *, allow_content_inequality: bool = False):
        """Creates a TensorInfo from a PyTorch tensor."""
        from torch.distributed.tensor import DTensor
        assert not isinstance(tensor, DTensor), 'Input tensor must not be a distributed tensor'
        assert tensor.is_contiguous(), 'Input tensor must be contiguous'
        numel: int = tensor.numel()
        data_ptr: int = tensor.data_ptr()
        dtype: DataType = DataType.from_torch_dtype(tensor.dtype)
        device_type: DeviceType = DeviceType.from_torch_device_type(tensor.device.type)
        return cls(name, data_ptr, numel=numel, dtype=dtype, device_type=device_type,
                   allow_content_inequality=allow_content_inequality)

    @classmethod
    def from_numpy(cls, tensor: 'np.ndarray', name: str, *, allow_content_inequality: bool = False):
        """Creates a TensorInfo from a Numpy tensor."""
        assert tensor.flags['C_CONTIGUOUS'], 'Input tensor must be contiguous'
        numel: int = tensor.size
        data_ptr: int = tensor.ctypes.data
        dtype: DataType = DataType.from_numpy_dtype(tensor.dtype)
        return cls(name, data_ptr, numel=numel, dtype=dtype, allow_content_inequality=allow_content_inequality,
                   device_type=DeviceType.CPU)


class SharedState:
    def __init__(self, tensor_infos: list[TensorInfo]):
        assert tensor_infos, 'At least one tensor info must be provided'
        self._infos = ffi.new('pcclTensorInfo_t[]', len(tensor_infos))
        self._name_strings = []
        for i, info in enumerate(tensor_infos):
            name_bytes = info.name.encode('utf-8')
            name_cstr = ffi.new('char[]', name_bytes)
            self._infos[i].name = name_cstr

            # Keep a reference to prevent the string from being freed; tie it to lifetime of SharedState python object
            self._name_strings.append(name_cstr)

            self._infos[i].data = ffi.cast('void*', info.data_ptr)
            self._infos[i].count = ffi.cast('size_t', info.numel)
            self._infos[i].datatype = ffi.cast('pcclDataType_t', info.dtype.value)
            self._infos[i].device_type = ffi.cast('pcclDeviceType_t', info.device_type.value)
            self._infos[i].allow_content_inequality = info.allow_content_inequality
        self._state = ffi.new('pcclSharedState_t*', {
            'revision': 0,
            'count': ffi.cast('size_t', len(tensor_infos)),
            'infos': self._infos,
        })

    @property
    def revision(self):
        return self._state[0].revision

    @revision.setter
    def revision(self, value: int):
        self._state[0].revision = value

    def push_revision(self):
        self._state[0].revision += 1


class SharedStateSyncInfo:
    def __init__(self, tx_bytes: int, rx_bytes: int):
        self.tx_bytes = tx_bytes
        self.rx_bytes = rx_bytes


class ReduceInfo:
    def __init__(self, local_world_size: int, tx_bytes: int, rx_bytes: int):
        self.local_world_size = local_world_size
        self.tx_bytes = tx_bytes
        self.rx_bytes = rx_bytes


class AsyncReduceHandle:
    def __init__(self, handle: ffi.CData):
        self._handle = handle
        self._info: Optional[Tuple[bool, int, ReduceInfo]] = None

    def wait(self) -> Tuple[bool, int, ReduceInfo]:
        """Awaits the completion of an async reduce operation. Blocks until the operation is complete."""
        if self._info is not None:
            return self._info

        info: ffi.CData = ffi.new('pcclReduceInfo_t*')
        status = C.pcclAwaitAsyncReduce(self._handle, info)
        is_success: bool = status == Result.SUCCESS.value
        self._info = (is_success, status, ReduceInfo(info.local_world_size, info.tx_bytes, info.rx_bytes))
        return self._info


def _create_ccoip_socket_address(address: Union[IPv4Address, IPv6Address], port: int) -> ffi.CData:
    socket_addr = ffi.new("ccoip_socket_address_t*")
    if isinstance(address, IPv4Address):
        socket_addr.inet.protocol = ffi.cast("ccoip_inet_protocol_t", C.inetIPv4)
        packed_ipv4 = address.packed
        for i, byte in enumerate(packed_ipv4):
            socket_addr.inet.ipv4.data[i] = byte & 255
    elif isinstance(address, IPv6Address):
        socket_addr.inet6.protocol = ffi.cast("ccoip_inet_protocol_t", C.inetIPv6)
        packed_ipv6 = address.packed
        for i, byte in enumerate(packed_ipv6):
            socket_addr.inet6.ipv6.data[i] = byte & 255
    else:
        raise ValueError(f'Unsupported IP address: {address}')
    socket_addr.port = port & 0xffff
    return socket_addr


class Communicator:
    """PCCL communicator."""

    def __init__(self, address: str, peer_group: int = 0):
        assert ":" in address, f'Invalid address: {address}, expected format: ip:port'
        params: ffi.CData = ffi.new('pcclCommCreateParams_t*')
        ip, port = address.split(":")
        ip = ip_address(ip)
        params.master_address = _create_ccoip_socket_address(ip, int(port))[0]
        params.peer_group = ffi.cast('uint32_t', peer_group)
        self._comm = ffi.new('pcclComm_t**')
        PCCLError.check(C.pcclCreateCommunicator(params, self._comm), "pcclCreateCommunicator")

    def __del__(self):
        C.pcclDestroyCommunicator(self._comm[0])

    def get_attribute(self, attribute: Attribute) -> int:
        """Get a communicator attribute."""
        value = ffi.new('int*')
        PCCLError.check(C.pcclGetAttribute(self._comm[0], attribute.value, value), "pcclGetAttribute")
        return value[0]

    def connect(self, n_attempts: int = 5):
        """
        Establishes a connection to a master node.
        Performs the specified number of attempts with a one second sleep interval.
        This function must be called on a communicator for the communicator to be usable.
        """
        for attempt in range(1, n_attempts + 1):
            try:
                PCCLError.check(C.pcclConnect(self._comm[0]), "pcclConnect")
                logging.info(f"Connected to the master node")
                break
            except PCCLError as e:
                logging.error(f"Failed to connect to the master node (Attempt {attempt}/{n_attempts}): {e}")
                time.sleep(1)
        else:
            raise Exception("Failed to connect to the master node")

    def update_topology(self):
        """
        Update the topology of a communicator if required.
        Topology updates are required when new peers join, in which case pcclUpdateTopology will
        automatically handle connection establishment with the new peer(s).
        Topology updates can also be triggered by the master node in response to bandwidth changes or other events.
        This function will block until the topology update is complete.
        """
        PCCLError.check(C.pcclUpdateTopology(self._comm[0]), "pcclUpdateTopology")

    def are_peers_pending(self) -> bool:
        """
        Check if there are any pending peer connections.
        If there are pending peers, it is recommended to call update_topology to establish connections.
        If there are no pending peers, invoking update_topology can be skipped without risking delaying peer acceptance.
        :return: True if there are pending peer connections, False otherwise.
        """
        pending = ffi.new('bool*')
        PCCLError.check(C.pcclArePeersPending(self._comm[0], pending), "pcclArePeersPending")
        return pending[0]

    def optimize_topology(self):
        """
        Optimize the topology of a communicator.
        This is recommended after the topology has changed after new peers join or leave.
        This function will block until the topology optimization is complete.
        """
        PCCLError.check(C.pcclOptimizeTopology(self._comm[0]), "pcclOptimizeTopology")

    def sync_shared_state(self, shared_state: SharedState) -> SharedStateSyncInfo:
        """
        Synchronizes the shared state between all peers that are currently accepted.
        If the shared state revision of this peer is outdated, the shared state will be updated.
        The function will not unblock until it is confirmed all peers have the same shared state revision.
        """
        sync_info: ffi.CData = ffi.new('pcclSharedStateSyncInfo_t*')
        PCCLError.check(C.pcclSynchronizeSharedState(self._comm[0], shared_state._state, sync_info),
                        "pcclSynchronizeSharedState")
        return SharedStateSyncInfo(sync_info.tx_bytes, sync_info.rx_bytes)

    def all_reduce(self, send: Union['torch.Tensor', 'np.ndarray'], recv: Union['torch.Tensor', 'np.ndarray'], *,
                   op: ReduceOp, tag: int = 0,
                   operand_descriptor: Optional[ReduceOperandDescriptor] = None,
                   quantization_options: Optional[QuantizationOptions] = None) -> ReduceInfo:
        """Performs an all reduce operation on a communicator. Blocks until the all reduce is complete."""
        if not isinstance(torch, ModuleDummy) and isinstance(send, torch.Tensor) and isinstance(recv, torch.Tensor):
            return self._all_reduce_pt(send, recv, op=op, tag=tag,
                                       operand_descriptor=operand_descriptor,
                                       quantization_options=quantization_options)
        elif not isinstance(np, ModuleDummy) and isinstance(send, np.ndarray) and isinstance(recv, np.ndarray):
            return self._all_reduce_np(send, recv, op=op, tag=tag,
                                       operand_descriptor=operand_descriptor,
                                       quantization_options=quantization_options)
        else:
            raise ValueError(
                f'Unsupported input types: {type(send)}, {type(recv)}; send and recv must either be both torch.Tensor or both np.ndarray')

    def _all_reduce_pt(self, send: 'torch.Tensor', recv: 'torch.Tensor', *, op: ReduceOp, tag: int,
                       operand_descriptor: Optional[ReduceOperandDescriptor],
                       quantization_options: Optional[QuantizationOptions]) -> ReduceInfo:
        assert send.is_contiguous(), 'Input tensor must be contiguous'
        assert recv.is_contiguous(), 'Output tensor must be contiguous'
        assert send.device == recv.device, 'Input and output tensors must be on the same device'
        assert send.dtype == recv.dtype, 'Input and output tensors must have the same dtype'
        assert send.device.type == 'cpu', 'Only CPU tensors are supported'
        assert send.numel() == recv.numel(), 'Input and output tensors must have the same number of elements'

        sendbuff: ffi.CData = ffi.cast('void*', send.data_ptr())
        recvbuff: ffi.CData = ffi.cast('void*', recv.data_ptr())
        num_elements: int = recv.numel()
        dtype: DataType = DataType.from_torch_dtype(send.dtype)

        if operand_descriptor is None:
            operand_descriptor = ReduceOperandDescriptor(
                datatype=dtype,
                distribution_hint=DistributionHint.NONE
            )

        if quantization_options is None:
            quantization_options = QuantizationOptions(
                quantized_datatype=dtype,
                algorithm=QuantizationAlgorithm.NONE
            )

        descriptor = ReduceDescriptor(
            count=num_elements,
            op=op,
            tag=tag,
            operand_descriptor=operand_descriptor,
            quantization_options=quantization_options
        ).to_c()

        info: ffi.CData = ffi.new('pcclReduceInfo_t*')
        PCCLError.check(
            C.pcclAllReduce(sendbuff, recvbuff, descriptor, self._comm[0], info),
            "pcclAllReduce"
        )
        return ReduceInfo(info.local_world_size, info.tx_bytes, info.rx_bytes)

    def _all_reduce_np(self, send: 'np.ndarray', recv: 'np.ndarray', *, op: ReduceOp, tag: int,
                       operand_descriptor: Optional[ReduceOperandDescriptor],
                       quantization_options: Optional[QuantizationOptions]) -> ReduceInfo:
        assert send.flags['C_CONTIGUOUS'], 'Input tensor must be contiguous'
        assert recv.flags['C_CONTIGUOUS'], 'Output tensor must be contiguous'
        assert send.dtype == recv.dtype, 'Input and output tensors must have the same dtype'
        assert send.size == recv.size, 'Input and output tensors must have the same number of elements'

        sendbuff: ffi.CData = ffi.cast('void*', send.ctypes.data)
        recvbuff: ffi.CData = ffi.cast('void*', recv.ctypes.data)
        num_elements: int = recv.size

        dtype: DataType = DataType.from_numpy_dtype(send.dtype)

        if operand_descriptor is None:
            operand_descriptor = ReduceOperandDescriptor(
                datatype=dtype,
                distribution_hint=DistributionHint.NONE
            )

        if quantization_options is None:
            quantization_options = QuantizationOptions(
                quantized_datatype=dtype,
                algorithm=QuantizationAlgorithm.NONE
            )

        descriptor = ReduceDescriptor(
            count=num_elements,
            op=op,
            tag=tag,
            operand_descriptor=operand_descriptor,
            quantization_options=quantization_options
        ).to_c()

        info: ffi.CData = ffi.new('pcclReduceInfo_t*')
        PCCLError.check(
            C.pcclAllReduce(sendbuff, recvbuff, descriptor, self._comm[0], info),
            "pcclAllReduce"
        )
        return ReduceInfo(info.local_world_size, info.tx_bytes, info.rx_bytes)

    def all_reduce_async(self, send: Union['torch.Tensor', 'np.ndarray'], recv: Union['torch.Tensor', 'np.ndarray'], *,
                         op: ReduceOp, tag: int = 0,
                         operand_descriptor: Optional[ReduceOperandDescriptor] = None,
                         quantization_options: Optional[QuantizationOptions] = None) -> AsyncReduceHandle:
        """Performs an all reduce operation on a communicator. Async version of all_reduce."""
        if not isinstance(torch, ModuleDummy) and isinstance(send, torch.Tensor) and isinstance(recv, torch.Tensor):
            return self._all_reduce_async_pt(send, recv, op=op, tag=tag,
                                             operand_descriptor=operand_descriptor,
                                             quantization_options=quantization_options)
        elif not isinstance(np, ModuleDummy) and isinstance(send, np.ndarray) and isinstance(recv, np.ndarray):
            return self._all_reduce_async_np(send, recv, op=op, tag=tag,
                                             operand_descriptor=operand_descriptor,
                                             quantization_options=quantization_options)
        else:
            raise ValueError(
                f'Unsupported input types: {type(send)}, {type(recv)}; send and recv must either be both torch.Tensor or both np.ndarray')

    def _all_reduce_async_pt(self, send: 'torch.Tensor', recv: 'torch.Tensor', *, op: ReduceOp, tag: int,
                             operand_descriptor: Optional[ReduceOperandDescriptor],
                             quantization_options: Optional[QuantizationOptions]) -> AsyncReduceHandle:
        assert send.is_contiguous(), 'Input tensor must be contiguous'
        assert recv.is_contiguous(), 'Output tensor must be contiguous'
        assert send.device == recv.device, 'Input and output tensors must be on the same device'
        assert send.dtype == recv.dtype, 'Input and output tensors must have the same dtype'
        assert send.device.type == 'cpu', 'Only CPU tensors are supported'
        assert send.numel() == recv.numel(), 'Input and output tensors must have the same number of elements'

        sendbuff: ffi.CData = ffi.cast('void*', send.data_ptr())
        recvbuff: ffi.CData = ffi.cast('void*', recv.data_ptr())
        num_elements: int = recv.numel()
        dtype: DataType = DataType.from_torch_dtype(send.dtype)

        if operand_descriptor is None:
            operand_descriptor = ReduceOperandDescriptor(
                datatype=dtype,
                distribution_hint=DistributionHint.NONE
            )

        if quantization_options is None:
            quantization_options = QuantizationOptions(
                quantized_datatype=dtype,
                algorithm=QuantizationAlgorithm.NONE
            )

        descriptor = ReduceDescriptor(
            count=num_elements,
            op=op,
            tag=tag,
            operand_descriptor=operand_descriptor,
            quantization_options=quantization_options
        ).to_c()

        handle: ffi.CData = ffi.new('pcclAsyncReduceOp_t*')
        PCCLError.check(
            C.pcclAllReduceAsync(sendbuff, recvbuff, descriptor, self._comm[0], handle),
            "pcclAllReduceAsync"
        )
        return AsyncReduceHandle(handle)

    def _all_reduce_async_np(self, send: 'np.ndarray', recv: 'np.ndarray', *, op: ReduceOp, tag: int,
                             operand_descriptor: Optional[ReduceOperandDescriptor],
                             quantization_options: Optional[QuantizationOptions]) -> AsyncReduceHandle:
        assert send.flags['C_CONTIGUOUS'], 'Input tensor must be contiguous'
        assert recv.flags['C_CONTIGUOUS'], 'Output tensor must be contiguous'
        assert send.dtype == recv.dtype, 'Input and output tensors must have the same dtype'
        assert send.size == recv.size, 'Input and output tensors must have the same number of elements'

        sendbuff: ffi.CData = ffi.cast('void*', send.ctypes.data)
        recvbuff: ffi.CData = ffi.cast('void*', recv.ctypes.data)
        num_elements: int = recv.size

        dtype: DataType = DataType.from_numpy_dtype(send.dtype)

        if operand_descriptor is None:
            operand_descriptor = ReduceOperandDescriptor(
                datatype=dtype,
                distribution_hint=DistributionHint.NONE
            )

        if quantization_options is None:
            quantization_options = QuantizationOptions(
                quantized_datatype=dtype,
                algorithm=QuantizationAlgorithm.NONE
            )

        descriptor = ReduceDescriptor(
            count=num_elements,
            op=op,
            tag=tag,
            operand_descriptor=operand_descriptor,
            quantization_options=quantization_options
        ).to_c()

        handle: ffi.CData = ffi.new('pcclAsyncReduceOp_t*')
        PCCLError.check(
            C.pcclAllReduceAsync(sendbuff, recvbuff, descriptor, self._comm[0], handle),
            "pcclAllReduceAsync"
        )
        return AsyncReduceHandle(handle)

    def all_reduce_multiple_with_retry(self, descriptors: List[ReduceOpDescriptor], *,
                                       max_in_flight: int = 4) -> ReduceInfo:
        """
        Performs multiple all reduces concurrently.
        If any of the all reduce operations fail, the function will await all outstanding operations and retry the failed ones.
        The function will not complete until all operations have completed successfully or the local world size has dropped below 2.
        @note Different reduce operations may have been performed with different local world sizes if peers dropped out during the operation.
        The local world size populated in the reduce info will be the local world size after all operations have completed. No veracity guarantees are made about this value beyond for heuristic usage.
        """
        descriptors_c = ffi.new('pcclReduceOpDescriptor_t[]', len(descriptors))
        for i, desc in enumerate(descriptors):
            desc.to_c_inpl(descriptors_c[i])

        info: ffi.CData = ffi.new('pcclReduceInfo_t*')
        PCCLError.check(
            C.pcclAllReduceMultipleWithRetry(descriptors_c, len(descriptors), self._comm[0], info, max_in_flight),
            "pcclAllReduceMultipleWithRetry"
        )
        return ReduceInfo(info.local_world_size, info.tx_bytes, info.rx_bytes)


class MasterNode:
    def __init__(self, listen_address: str):
        assert ":" in listen_address, f'Invalid listen address: {listen_address}, expected format: ip:port'
        ip, port = listen_address.split(":")
        ip = ip_address(ip)
        ccoip_address = _create_ccoip_socket_address(ip, int(port))
        self._socket_address = ccoip_address
        self._master = ffi.new('pcclMasterInstance_t**')
        PCCLError.check(C.pcclCreateMaster(ccoip_address[0], self._master), "pcclCreateMaster")

    def run(self):
        """Runs a master node. This function is non-blocking."""
        PCCLError.check(C.pcclRunMaster(self._master[0]), "pcclRunMaster")

    def interrupt(self):
        """Interrupts a master node."""
        PCCLError.check(C.pcclInterruptMaster(self._master[0]), "pcclInterruptMaster")

    def __del__(self):
        PCCLError.check(C.pcclMasterAwaitTermination(self._master[0]), "pcclMasterAwaitTermination")
        PCCLError.check(C.pcclDestroyMaster(self._master[0]), "pcclDestroyMaster")
