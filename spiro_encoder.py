#!/usr/bin/env python3
"""
SpiroEncoder.py - Encode x,y spirometry data for JavaScript
Converts [x, y] pairs into packed 32-bit values for efficient WebSocket transmission
Compatible with spiro_decoder.js on JavaScript side
"""

import numpy as np
from typing import List, Tuple, Dict, Union

# ============ ENCODING FUNCTIONS ============

def encode_spiro_value(volume: Union[int, float], flow: Union[int, float]) -> int:
    """
    Encode volume (x) and flow (y) into single 32-bit value
    
    Format:
    - Upper 16 bits (bits 16-31): volume (x-axis)
    - Lower 16 bits (bits 0-15): flow (y-axis)
    
    Args:
        volume: X-axis value (0-65535)
        flow: Y-axis value (0-65535)
    
    Returns:
        Encoded 32-bit integer
    
    Example:
        >>> encode_spiro_value(128, 512)
        8438912
    """
    # Ensure values are in 16-bit range
    x = int(volume) & 0xFFFF
    y = int(flow) & 0xFFFF
    
    # Pack: upper 16 bits = x, lower 16 bits = y
    encoded = (x << 16) | y
    
    return encoded


def encode_spiro_array(x_array: List[Union[int, float]], 
                       y_array: List[Union[int, float]]) -> List[int]:
    """
    Encode array of x,y pairs into packed values
    
    Args:
        x_array: Array of x values
        y_array: Array of y values (must be same length as x_array)
    
    Returns:
        Array of encoded 32-bit integers
    
    Raises:
        ValueError: If arrays are different lengths
    
    Example:
        >>> encode_spiro_array([128, 128, 129], [512, 768, 0])
        [8438912, 8439040, 8454144]
    """
    if len(x_array) != len(y_array):
        raise ValueError(f"Array length mismatch: x={len(x_array)}, y={len(y_array)}")
    
    encoded = []
    for i in range(len(x_array)):
        encoded_value = encode_spiro_value(x_array[i], y_array[i])
        encoded.append(encoded_value)
    
    return encoded


# ============ NORMALIZATION FUNCTIONS ============

def normalize_to_16bit(values: Union[List, np.ndarray], 
                      min_val: float = None, 
                      max_val: float = None) -> np.ndarray:
    """
    Normalize values to 0-65535 range (16-bit unsigned integer range)
    
    Args:
        values: Array of numeric values
        min_val: Minimum value (auto-detected if None)
        max_val: Maximum value (auto-detected if None)
    
    Returns:
        Array of integers in range [0, 65535]
    
    Example:
        >>> normalize_to_16bit([100.5, 102.3, 98.7])
        array([42598, 54512, 28321])
    """
    arr = np.array(values, dtype=float)
    
    # Auto-detect min/max if not provided
    if min_val is None:
        min_val = np.min(arr)
    if max_val is None:
        max_val = np.max(arr)
    
    # Avoid division by zero
    if max_val == min_val:
        return np.zeros(len(arr), dtype=int)
    
    # Normalize to [0, 1], then to [0, 65535]
    normalized = (arr - min_val) / (max_val - min_val)
    scaled = (normalized * 65535).astype(int)
    
    return scaled


def normalize_pair_arrays(x_array: Union[List, np.ndarray],
                         y_array: Union[List, np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Normalize both x and y arrays independently
    
    Args:
        x_array: X values
        y_array: Y values
    
    Returns:
        Tuple of (normalized_x, normalized_y)
    
    Example:
        >>> x_norm, y_norm = normalize_pair_arrays([0, 0.005, 0.010], [100, 102, 99])
        >>> x_norm
        array([    0, 32767, 65535])
        >>> y_norm
        array([32767, 65535,     0])
    """
    x_norm = normalize_to_16bit(x_array)
    y_norm = normalize_to_16bit(y_array)
    
    return x_norm, y_norm


# ============ SPIROMETRY-SPECIFIC FUNCTIONS ============

def prepare_spiro_data(x_array: Union[List, np.ndarray],
                      y_array: Union[List, np.ndarray],
                      normalize: bool = True) -> List[int]:
    """
    Prepare spirometry data for transmission
    
    Args:
        x_array: Volume/Time values
        y_array: Flow/Pressure values
        normalize: Whether to normalize to 16-bit (default True)
    
    Returns:
        Array of encoded packed values
    
    Example:
        >>> x = [0.0, 0.005, 0.010]
        >>> y = [100.5, 102.3, 98.7]
        >>> encoded = prepare_spiro_data(x, y)
        >>> encoded
        [8438464, 8438912, 8440192]
    """
    if normalize:
        x_norm, y_norm = normalize_pair_arrays(x_array, y_array)
        return encode_spiro_array(x_norm, y_norm)
    else:
        # Use as-is (must already be in 0-65535 range)
        return encode_spiro_array(x_array, y_array)


def split_encoded_data(encoded_array: List[int], 
                       chunk_size: int = 100) -> List[List[int]]:
    """
    Split large encoded array into chunks for streaming
    
    Args:
        encoded_array: Array of encoded values
        chunk_size: Maximum size of each chunk
    
    Returns:
        List of smaller arrays
    
    Example:
        >>> arr = list(range(250))
        >>> chunks = split_encoded_data(arr, chunk_size=100)
        >>> len(chunks)
        3
    """
    chunks = []
    for i in range(0, len(encoded_array), chunk_size):
        chunks.append(encoded_array[i:i+chunk_size])
    
    return chunks


# ============ METADATA FUNCTIONS ============

def get_encoded_data_stats(encoded_array: List[int]) -> Dict:
    """
    Get statistics about encoded data
    
    Args:
        encoded_array: Array of encoded values
    
    Returns:
        Dictionary with statistics
    
    Example:
        >>> stats = get_encoded_data_stats([8438464, 8438912])
        >>> stats['count']
        2
    """
    if not encoded_array:
        return {
            'count': 0,
            'min_x': None,
            'max_x': None,
            'min_y': None,
            'max_y': None
        }
    
    x_values = [(v >> 16) & 0xFFFF for v in encoded_array]
    y_values = [v & 0xFFFF for v in encoded_array]
    
    return {
        'count': len(encoded_array),
        'min_x': min(x_values),
        'max_x': max(x_values),
        'min_y': min(y_values),
        'max_y': max(y_values),
        'avg_x': sum(x_values) / len(x_values),
        'avg_y': sum(y_values) / len(y_values)
    }


# ============ WEBSOCKET INTEGRATION ============

def create_spiro_message(encoded_array: List[int], msg_type: str = "createSpiroLiveGraph") -> str:
    """
    Create WebSocket message with encoded data
    
    Args:
        encoded_array: Array of encoded values
        msg_type: Message type (default: "createSpiroLiveGraph")
    
    Returns:
        Formatted message string: "msgType~[encoded_data]"
    
    Example:
        >>> msg = create_spiro_message([8438464, 8438912])
        >>> msg
        'createSpiroLiveGraph~[8438464, 8438912]'
    """
    import json
    array_json = json.dumps(encoded_array)
    return f"{msg_type}~{array_json}"


# ============ STREAMING ENCODER ============

class SpiroStreamEncoder:
    """
    Real-time streaming encoder for continuous spirometry data
    """
    
    def __init__(self, chunk_size: int = 100):
        """
        Initialize encoder
        
        Args:
            chunk_size: Number of values to buffer before sending
        """
        self.chunk_size = chunk_size
        self.x_buffer = []
        self.y_buffer = []
        self.message_count = 0
    
    def add_value(self, x: Union[int, float], y: Union[int, float]) -> bool:
        """
        Add single x,y value
        
        Returns:
            True if chunk is full and ready to send
        """
        self.x_buffer.append(x)
        self.y_buffer.append(y)
        
        if len(self.x_buffer) >= self.chunk_size:
            return True
        
        return False
    
    def add_array(self, x_array: List, y_array: List) -> bool:
        """
        Add array of x,y values
        
        Returns:
            True if chunk is full and ready to send
        """
        self.x_buffer.extend(x_array)
        self.y_buffer.extend(y_array)
        
        if len(self.x_buffer) >= self.chunk_size:
            return True
        
        return False
    
    def get_encoded_chunk(self, normalize: bool = True) -> List[int]:
        """
        Get encoded chunk and reset buffers
        
        Args:
            normalize: Whether to normalize to 16-bit
        
        Returns:
            Array of encoded values (up to chunk_size length)
        """
        # Extract chunk
        x_chunk = self.x_buffer[:self.chunk_size]
        y_chunk = self.y_buffer[:self.chunk_size]
        
        # Remove from buffer
        self.x_buffer = self.x_buffer[self.chunk_size:]
        self.y_buffer = self.y_buffer[self.chunk_size:]
        
        # Encode
        encoded = prepare_spiro_data(x_chunk, y_chunk, normalize=normalize)
        self.message_count += 1
        
        return encoded
    
    def has_data(self) -> bool:
        """Check if buffer has data"""
        return len(self.x_buffer) > 0
    
    def flush(self, normalize: bool = True) -> List[int]:
        """
        Get all remaining data (even if < chunk_size)
        
        Returns:
            Array of encoded values
        """
        if not self.has_data():
            return []
        
        encoded = prepare_spiro_data(self.x_buffer, self.y_buffer, normalize=normalize)
        self.x_buffer = []
        self.y_buffer = []
        
        return encoded


# ============ USAGE EXAMPLE ============

if __name__ == "__main__":
    print("=== Spirometry Encoder Examples ===\n")
    
    # Example 1: Simple encoding
    print("Example 1: Encode single pair")
    x, y = 128, 512
    encoded = encode_spiro_value(x, y)
    print(f"  encode_spiro_value({x}, {y}) = {encoded}\n")
    
    # Example 2: Encode array
    print("Example 2: Encode array of pairs")
    x_arr = [128, 128, 129, 129, 129]
    y_arr = [512, 768, 0, 128, 896]
    encoded_arr = encode_spiro_array(x_arr, y_arr)
    print(f"  Input: x={x_arr}")
    print(f"         y={y_arr}")
    print(f"  Output: {encoded_arr}\n")
    
    # Example 3: Normalize and encode
    print("Example 3: Normalize and encode real data")
    x_real = [0.0, 0.005, 0.010, 0.015, 0.020]
    y_real = [100.5, 102.3, 98.7, 105.2, 103.1]
    encoded_real = prepare_spiro_data(x_real, y_real, normalize=True)
    print(f"  Input: x={x_real}")
    print(f"         y={y_real}")
    print(f"  Encoded: {encoded_real}\n")
    
    # Example 4: Statistics
    print("Example 4: Get statistics")
    stats = get_encoded_data_stats(encoded_real)
    print(f"  Count: {stats['count']}")
    print(f"  X range: {stats['min_x']} - {stats['max_x']}")
    print(f"  Y range: {stats['min_y']} - {stats['max_y']}\n")
    
    # Example 5: Streaming encoder
    print("Example 5: Streaming encoder")
    encoder = SpiroStreamEncoder(chunk_size=3)
    
    # Add values one by one
    for x, y in zip(x_real, y_real):
        should_send = encoder.add_value(x, y)
        if should_send:
            chunk = encoder.get_encoded_chunk()
            print(f"  📤 Chunk {encoder.message_count}: {chunk}")
    
    # Flush remaining
    if encoder.has_data():
        chunk = encoder.flush()
        print(f"  📤 Final chunk {encoder.message_count + 1}: {chunk}\n")
    
    # Example 6: Create WebSocket message
    print("Example 6: Create WebSocket message")
    msg = create_spiro_message(encoded_real)
    print(f"  {msg[:60]}...\n")
    
    print("All examples complete!")

def encode_spiro_triple(flow: float, volume: float, vt_volume: float) -> int:
    """
    Encode FV_Flow(16bit), FV_Volume(8bit), VT_Volume(8bit) into 32-bit value
    
    Format:
    - Bits 16-31: flow     (FV_Flow_Lps)
    - Bits  8-15: volume   (FV_Volume_L)  
    - Bits  0-7:  vt_vol   (VT_Volume_Exhale_L)
    """
    # Clamp to ranges
    flow_int     = int(flow) & 0xFFFF      # 16 bits
    volume_int   = int(volume * 10) & 0xFF # 8 bits (x10 for precision)
    vt_vol_int   = int(vt_volume * 10) & 0xFF # 8 bits (x10 for precision)
    
    # Pack: flow(16) | volume(8) | vt_vol(8)
    encoded = (flow_int << 16) | (volume_int << 8) | vt_vol_int
    return encoded


def prepare_spiro_triple(flow_array: List[float], 
                        volume_array: List[float], 
                        vt_array: List[float]) -> List[int]:
    """
    Encode 3 parallel arrays into single packed array
    """
    if not (len(flow_array) == len(volume_array) == len(vt_array)):
        raise ValueError("All arrays must be same length")
    
    # Filter None/empty values (align sparse data)
    valid_indices = [i for i, (f,v,vt) in enumerate(zip(flow_array, volume_array, vt_array)) 
                    if f is not None and v is not None and vt is not None]
    
    if not valid_indices:
        return []
    
    # Extract valid triples only
    flow_valid = [flow_array[i] for i in valid_indices]
    vol_valid  = [volume_array[i] for i in valid_indices]  
    vt_valid   = [vt_array[i] for i in valid_indices]
    
    # Normalize each independently to fit ranges
    flow_norm  = normalize_to_16bit(flow_valid, 0, 15)  # Flow: 0-15 L/s
    vol_norm   = normalize_to_16bit(vol_valid, 0, 8)    # Vol: 0-8L → 0-80
    vt_norm    = normalize_to_16bit(vt_valid, 0, 8)     # VT: 0-8L → 0-80
    
    # Pack into single values
    encoded = []
    for f, v, vt in zip(flow_norm, vol_norm, vt_norm):
        encoded.append(encode_spiro_triple(f/65535*15, v/65535*8, vt/65535*8))
    
    return encoded
