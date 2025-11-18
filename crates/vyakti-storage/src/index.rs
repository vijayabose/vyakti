//! Index file format and serialization.

use memmap2::{Mmap, MmapOptions};
use serde::{Deserialize, Serialize};
use std::fs::File;
use std::io::{Read, Write};
use std::path::Path;
use vyakti_common::{Result, VyaktiError};

/// Index header structure
#[repr(C)]
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct IndexHeader {
    /// Magic bytes: "VYAK"
    pub magic: [u8; 4],
    /// Format version
    pub version: u32,
    /// Number of vectors
    pub num_vectors: u64,
    /// Vector dimension
    pub dimension: u32,
    /// Backend type identifier
    pub backend_type: u32,
    /// Flags (compact, recompute, etc.)
    pub flags: u64,
}

impl IndexHeader {
    /// Magic bytes constant
    pub const MAGIC: &'static [u8; 4] = b"VYAK";

    /// Current format version
    pub const VERSION: u32 = 1;

    /// Create a new index header.
    ///
    /// # Arguments
    ///
    /// * `num_vectors` - Number of vectors in the index
    /// * `dimension` - Dimensionality of vectors
    ///
    /// # Returns
    ///
    /// A new index header with default values
    pub fn new(num_vectors: usize, dimension: usize) -> Self {
        Self {
            magic: *Self::MAGIC,
            version: Self::VERSION,
            num_vectors: num_vectors as u64,
            dimension: dimension as u32,
            backend_type: 0,
            flags: 0,
        }
    }

    /// Validate magic bytes.
    ///
    /// # Returns
    ///
    /// `true` if magic bytes are valid, `false` otherwise
    pub fn is_valid(&self) -> bool {
        &self.magic == Self::MAGIC
    }

    /// Validate version.
    ///
    /// # Returns
    ///
    /// `true` if version is supported, `false` otherwise
    pub fn is_version_supported(&self) -> bool {
        self.version == Self::VERSION
    }

    /// Validate the entire header.
    ///
    /// # Returns
    ///
    /// `Ok(())` if valid, error otherwise
    pub fn validate(&self) -> Result<()> {
        if !self.is_valid() {
            return Err(VyaktiError::Storage(
                "Invalid magic bytes in index header".to_string(),
            ));
        }

        if !self.is_version_supported() {
            return Err(VyaktiError::Storage(format!(
                "Unsupported index version: {}",
                self.version
            )));
        }

        if self.dimension == 0 {
            return Err(VyaktiError::Storage(
                "Index dimension cannot be zero".to_string(),
            ));
        }

        Ok(())
    }

    /// Get header size in bytes.
    pub fn size() -> usize {
        std::mem::size_of::<Self>()
    }

    /// Write header to a file.
    ///
    /// # Arguments
    ///
    /// * `file` - File to write to
    pub fn write_to<W: Write>(&self, writer: &mut W) -> Result<()> {
        bincode::serialize_into(writer, self)
            .map_err(|e| VyaktiError::Serialization(format!("Failed to write header: {}", e)))
    }

    /// Read header from a file.
    ///
    /// # Arguments
    ///
    /// * `reader` - Reader to read from
    pub fn read_from<R: Read>(reader: &mut R) -> Result<Self> {
        let header: IndexHeader = bincode::deserialize_from(reader)
            .map_err(|e| VyaktiError::Serialization(format!("Failed to read header: {}", e)))?;

        header.validate()?;
        Ok(header)
    }
}

/// Memory-mapped index file.
#[derive(Debug)]
pub struct MmapIndex {
    _mmap: Mmap,
    header: IndexHeader,
}

impl MmapIndex {
    /// Load an index file using memory mapping.
    ///
    /// # Arguments
    ///
    /// * `path` - Path to the index file
    ///
    /// # Returns
    ///
    /// Memory-mapped index
    pub fn load<P: AsRef<Path>>(path: P) -> Result<Self> {
        let file = File::open(path.as_ref())
            .map_err(|e| VyaktiError::Storage(format!("Failed to open index file: {}", e)))?;

        let mmap = unsafe {
            MmapOptions::new()
                .map(&file)
                .map_err(|e| VyaktiError::Storage(format!("Failed to memory-map file: {}", e)))?
        };

        // Read header from the beginning of the file
        if mmap.len() < IndexHeader::size() {
            return Err(VyaktiError::Storage(
                "Index file too small to contain header".to_string(),
            ));
        }

        let header: IndexHeader =
            bincode::deserialize(&mmap[..IndexHeader::size()]).map_err(|e| {
                VyaktiError::Serialization(format!("Failed to deserialize header: {}", e))
            })?;

        header.validate()?;

        Ok(Self {
            _mmap: mmap,
            header,
        })
    }

    /// Get the index header.
    pub fn header(&self) -> &IndexHeader {
        &self.header
    }

    /// Get raw data slice (excluding header).
    pub fn data(&self) -> &[u8] {
        &self._mmap[IndexHeader::size()..]
    }
}

/// Index writer for creating index files.
pub struct IndexWriter {
    file: File,
    header: IndexHeader,
}

impl std::fmt::Debug for IndexWriter {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("IndexWriter")
            .field("header", &self.header)
            .finish()
    }
}

impl IndexWriter {
    /// Create a new index writer.
    ///
    /// # Arguments
    ///
    /// * `path` - Path to create the index file
    /// * `num_vectors` - Number of vectors
    /// * `dimension` - Vector dimension
    pub fn create<P: AsRef<Path>>(path: P, num_vectors: usize, dimension: usize) -> Result<Self> {
        let mut file = File::create(path.as_ref())
            .map_err(|e| VyaktiError::Storage(format!("Failed to create index file: {}", e)))?;

        let header = IndexHeader::new(num_vectors, dimension);
        header.write_to(&mut file)?;

        Ok(Self { file, header })
    }

    /// Write data to the index file.
    ///
    /// # Arguments
    ///
    /// * `data` - Data to write
    pub fn write(&mut self, data: &[u8]) -> Result<()> {
        self.file
            .write_all(data)
            .map_err(|e| VyaktiError::Storage(format!("Failed to write data: {}", e)))
    }

    /// Finalize the index file.
    pub fn finalize(self) -> Result<()> {
        self.file
            .sync_all()
            .map_err(|e| VyaktiError::Storage(format!("Failed to sync file: {}", e)))
    }

    /// Get the header.
    pub fn header(&self) -> &IndexHeader {
        &self.header
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;
    use tempfile::tempdir;

    #[test]
    fn test_index_header_new() {
        let header = IndexHeader::new(1000, 384);
        assert_eq!(header.magic, *IndexHeader::MAGIC);
        assert_eq!(header.version, IndexHeader::VERSION);
        assert_eq!(header.num_vectors, 1000);
        assert_eq!(header.dimension, 384);
    }

    #[test]
    fn test_index_header_validation() {
        let header = IndexHeader::new(100, 128);
        assert!(header.is_valid());
        assert!(header.is_version_supported());
        assert!(header.validate().is_ok());
    }

    #[test]
    fn test_index_header_invalid_magic() {
        let mut header = IndexHeader::new(100, 128);
        header.magic = *b"XXXX";
        assert!(!header.is_valid());
        assert!(header.validate().is_err());
    }

    #[test]
    fn test_index_header_invalid_version() {
        let mut header = IndexHeader::new(100, 128);
        header.version = 999;
        assert!(!header.is_version_supported());
        assert!(header.validate().is_err());
    }

    #[test]
    fn test_index_header_zero_dimension() {
        let mut header = IndexHeader::new(100, 0);
        header.dimension = 0;
        assert!(header.validate().is_err());
    }

    #[test]
    fn test_index_header_serialization() {
        let header = IndexHeader::new(500, 256);
        let mut buffer = Vec::new();

        header.write_to(&mut buffer).unwrap();
        assert!(!buffer.is_empty());

        let mut cursor = Cursor::new(buffer);
        let deserialized = IndexHeader::read_from(&mut cursor).unwrap();

        assert_eq!(deserialized.num_vectors, header.num_vectors);
        assert_eq!(deserialized.dimension, header.dimension);
    }

    #[test]
    fn test_index_writer_create() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("test.idx");

        let mut writer = IndexWriter::create(&path, 100, 384).unwrap();
        assert_eq!(writer.header().num_vectors, 100);
        assert_eq!(writer.header().dimension, 384);

        let data = vec![1u8, 2, 3, 4, 5];
        writer.write(&data).unwrap();
        writer.finalize().unwrap();

        assert!(path.exists());
    }

    #[test]
    fn test_mmap_index_load() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("test.idx");

        // Create an index file
        let mut writer = IndexWriter::create(&path, 50, 128).unwrap();
        let test_data = vec![42u8; 100];
        writer.write(&test_data).unwrap();
        writer.finalize().unwrap();

        // Load it with mmap
        let mmap_index = MmapIndex::load(&path).unwrap();
        assert_eq!(mmap_index.header().num_vectors, 50);
        assert_eq!(mmap_index.header().dimension, 128);
        assert_eq!(mmap_index.data().len(), 100);
        assert_eq!(mmap_index.data()[0], 42);
    }

    #[test]
    fn test_mmap_index_invalid_file() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("nonexistent.idx");

        let result = MmapIndex::load(&path);
        assert!(result.is_err());
    }

    #[test]
    fn test_mmap_index_corrupted_header() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("corrupted.idx");

        // Create a file with invalid header
        let mut file = File::create(&path).unwrap();
        file.write_all(b"INVALID_DATA").unwrap();

        let result = MmapIndex::load(&path);
        assert!(result.is_err());
    }

    #[test]
    fn test_index_header_size() {
        let size = IndexHeader::size();
        assert!(size > 0);
        assert!(size < 1024); // Should be reasonably small
    }

    #[test]
    fn test_roundtrip_write_read() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("roundtrip.idx");

        // Write
        let mut writer = IndexWriter::create(&path, 200, 512).unwrap();
        let original_data = vec![1u8, 2, 3, 4, 5, 6, 7, 8];
        writer.write(&original_data).unwrap();
        writer.finalize().unwrap();

        // Read back
        let mmap_index = MmapIndex::load(&path).unwrap();
        assert_eq!(mmap_index.header().num_vectors, 200);
        assert_eq!(mmap_index.header().dimension, 512);
        assert_eq!(mmap_index.data(), &original_data[..]);
    }

    #[test]
    fn test_mmap_index_too_small() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("too_small.idx");

        // Create a file that's too small to contain a header
        let mut file = File::create(&path).unwrap();
        file.write_all(b"X").unwrap();

        let result = MmapIndex::load(&path);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("too small"));
    }

    #[test]
    fn test_index_writer_create_invalid_path() {
        // Try to create a file in a directory that doesn't exist
        let result = IndexWriter::create("/nonexistent/dir/test.idx", 100, 384);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("Failed to create"));
    }

    #[test]
    fn test_index_writer_debug() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("debug.idx");
        let writer = IndexWriter::create(&path, 100, 256).unwrap();

        let debug_str = format!("{:?}", writer);
        assert!(debug_str.contains("IndexWriter"));
        assert!(debug_str.contains("header"));
    }

    #[test]
    fn test_mmap_index_debug() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("debug.idx");

        let mut writer = IndexWriter::create(&path, 50, 128).unwrap();
        writer.write(&[1u8; 10]).unwrap();
        writer.finalize().unwrap();

        let mmap_index = MmapIndex::load(&path).unwrap();
        let debug_str = format!("{:?}", mmap_index);
        assert!(debug_str.contains("MmapIndex"));
    }
}
