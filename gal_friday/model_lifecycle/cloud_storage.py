"""Cloud storage backends for model artifacts."""

from abc import ABC, abstractmethod
from pathlib import Path
import asyncio
from typing import Any
import aiofiles # type: ignore[import-untyped]
import hashlib

from gal_friday.config_manager import ConfigManager
from gal_friday.logger_service import LoggerService


class CloudStorageBackend(ABC):
    """Abstract base class for cloud storage backends."""
    
    @abstractmethod
    async def upload(self, local_path: Path, remote_path: str) -> bool:
        """Upload file or directory to cloud storage."""
        pass
    
    @abstractmethod
    async def download(self, remote_path: str, local_path: Path) -> bool:
        """Download file or directory from cloud storage."""
        pass
    
    @abstractmethod
    async def delete(self, remote_path: str) -> bool:
        """Delete file or directory from cloud storage."""
        pass
    
    @abstractmethod
    async def exists(self, remote_path: str) -> bool:
        """Check if file or directory exists in cloud storage."""
        pass


class GCSBackend(CloudStorageBackend):
    """Google Cloud Storage backend implementation."""
    
    def __init__(self, config: ConfigManager, logger: LoggerService) -> None:
        """Initialize GCS backend.
        
        Args:
            config: Configuration manager
            logger: Logger service
        """
        self.config = config
        self.logger = logger
        self._source_module = self.__class__.__name__
        
        # GCS configuration
        self.bucket_name = config.get("model_registry.gcs_bucket")
        self.project_id = config.get("gcp.project_id")
        
        # Initialize client
        self._init_client()
    
    def _init_client(self) -> None:
        """Initialize GCS client."""
        try:
            from google.cloud import storage # type: ignore[import-not-found]
            self.client = storage.Client(project=self.project_id)
            self.bucket = self.client.bucket(self.bucket_name)
            
            self.logger.info(
                f"Initialized GCS backend for bucket: {self.bucket_name}",
                source_module=self._source_module
            )
        except ImportError:
            raise ImportError("google-cloud-storage not installed")
        except Exception as e:
            raise RuntimeError(f"Failed to initialize GCS client: {e}")
    
    async def upload(self, local_path: Path, remote_path: str) -> bool:
        """Upload to GCS with progress tracking and verification."""
        try:
            if local_path.is_file():
                return await self._upload_file(local_path, remote_path)
            else:
                return await self._upload_directory(local_path, remote_path)
        except Exception:
            self.logger.exception(
                f"Failed to upload to GCS: {remote_path}",
                source_module=self._source_module
            )
            return False
    
    async def _upload_file(self, local_path: Path, remote_path: str) -> bool:
        """Upload single file to GCS."""
        try:
            # Calculate checksum
            local_checksum = await self._calculate_checksum(local_path)
            
            # Upload file
            blob = self.bucket.blob(remote_path)
            
            # Use async file reading
            async with aiofiles.open(local_path, 'rb') as f:
                content = await f.read()
                
            # Upload in thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(
                None,
                blob.upload_from_string,
                content,
                'application/octet-stream'
            )
            
            # Set metadata
            blob.metadata = {
                'checksum': local_checksum,
                'original_path': str(local_path)
            }
            await loop.run_in_executor(None, blob.patch)
            
            # Verify upload
            await loop.run_in_executor(None, blob.reload)
            remote_checksum = blob.metadata.get('checksum')
            
            if remote_checksum != local_checksum:
                self.logger.error(
                    f"Checksum mismatch after upload: {remote_path}",
                    source_module=self._source_module
                )
                return False
            
            self.logger.info(
                f"Successfully uploaded file to GCS: {remote_path}",
                source_module=self._source_module,
                context={
                    "size_bytes": local_path.stat().st_size,
                    "checksum": local_checksum
                }
            )
            return True
            
        except Exception:
            self.logger.exception(
                f"Failed to upload file to GCS: {remote_path}",
                source_module=self._source_module
            )
            return False
    
    async def _upload_directory(self, local_path: Path, remote_path: str) -> bool:
        """Upload directory recursively to GCS."""
        try:
            upload_tasks = []
            
            for file_path in local_path.rglob("*"):
                if file_path.is_file():
                    relative_path = file_path.relative_to(local_path)
                    remote_file_path = f"{remote_path}/{relative_path}".replace("\\", "/")
                    
                    task = self._upload_file(file_path, remote_file_path)
                    upload_tasks.append(task)
            
            results = await asyncio.gather(*upload_tasks, return_exceptions=True)
            
            # Check results
            failed_count = sum(1 for r in results if isinstance(r, Exception) or not r)
            if failed_count > 0:
                self.logger.error(
                    f"Failed to upload {failed_count}/{len(results)} files",
                    source_module=self._source_module
                )
                return False
            
            self.logger.info(
                f"Successfully uploaded directory to GCS: {remote_path}",
                source_module=self._source_module,
                context={"file_count": len(results)}
            )
            return True
            
        except Exception:
            self.logger.exception(
                f"Failed to upload directory to GCS: {remote_path}",
                source_module=self._source_module
            )
            return False
    
    async def download(self, remote_path: str, local_path: Path) -> bool:
        """Download from GCS."""
        try:
            # List all blobs with prefix
            loop = asyncio.get_event_loop()
            blobs = await loop.run_in_executor(
                None,
                list,
                self.bucket.list_blobs(prefix=remote_path)
            )
            
            if not blobs:
                self.logger.error(
                    f"No files found in GCS: {remote_path}",
                    source_module=self._source_module
                )
                return False
            
            # Download all blobs
            download_tasks = []
            for blob in blobs:
                # Calculate local file path
                relative_path = blob.name[len(remote_path):].lstrip("/")
                if not relative_path:  # Single file
                    file_path = local_path
                else:
                    file_path = local_path / relative_path
                
                task = self._download_file(blob, file_path)
                download_tasks.append(task)
            
            results = await asyncio.gather(*download_tasks, return_exceptions=True)
            
            # Check results
            failed_count = sum(1 for r in results if isinstance(r, Exception) or not r)
            if failed_count > 0:
                self.logger.error(
                    f"Failed to download {failed_count}/{len(results)} files",
                    source_module=self._source_module
                )
                return False
            
            return True
            
        except Exception:
            self.logger.exception(
                f"Failed to download from GCS: {remote_path}",
                source_module=self._source_module
            )
            return False
    
    async def _download_file(self, blob: Any, local_path: Path) -> bool:
        """Download single file from GCS."""
        try:
            # Create parent directory
            local_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Download to bytes
            loop = asyncio.get_event_loop()
            content = await loop.run_in_executor(None, blob.download_as_bytes)
            
            # Write async
            async with aiofiles.open(local_path, 'wb') as f:
                await f.write(content)
            
            # Verify checksum if available
            if blob.metadata and 'checksum' in blob.metadata:
                local_checksum = await self._calculate_checksum(local_path)
                if local_checksum != blob.metadata['checksum']:
                    self.logger.error(
                        f"Checksum mismatch after download: {local_path}",
                        source_module=self._source_module
                    )
                    return False
            
            return True
            
        except Exception:
            self.logger.exception(
                f"Failed to download file from GCS: {blob.name}",
                source_module=self._source_module
            )
            return False
    
    async def delete(self, remote_path: str) -> bool:
        """Delete from GCS."""
        try:
            loop = asyncio.get_event_loop()
            
            # Delete all blobs with prefix
            blobs = await loop.run_in_executor(
                None,
                list,
                self.bucket.list_blobs(prefix=remote_path)
            )
            
            for blob in blobs:
                await loop.run_in_executor(None, blob.delete)
            
            self.logger.info(
                f"Deleted {len(blobs)} files from GCS: {remote_path}",
                source_module=self._source_module
            )
            return True
            
        except Exception:
            self.logger.exception(
                f"Failed to delete from GCS: {remote_path}",
                source_module=self._source_module
            )
            return False
    
    async def exists(self, remote_path: str) -> bool:
        """Check if exists in GCS."""
        try:
            blob = self.bucket.blob(remote_path)
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(None, blob.exists)
        except Exception:
            return False
    
    async def _calculate_checksum(self, file_path: Path) -> str:
        """Calculate SHA256 checksum of file."""
        sha256_hash = hashlib.sha256()
        
        async with aiofiles.open(file_path, "rb") as f:
            while chunk := await f.read(8192):
                sha256_hash.update(chunk)
        
        return sha256_hash.hexdigest()


class S3Backend(CloudStorageBackend):
    """AWS S3 backend implementation."""
    
    def __init__(self, config: ConfigManager, logger: LoggerService) -> None:
        """Initialize S3 backend."""
        self.config = config
        self.logger = logger
        self._source_module = self.__class__.__name__
        
        # S3 configuration
        self.bucket_name = config.get("model_registry.s3_bucket")
        self.region = config.get("aws.region", "us-east-1")
        
        # Initialize client
        self._init_client()
    
    def _init_client(self) -> None:
        """Initialize S3 client."""
        try:
            import aioboto3 # type: ignore[import-untyped]
            self.session = aioboto3.Session()
            
            self.logger.info(
                f"Initialized S3 backend for bucket: {self.bucket_name}",
                source_module=self._source_module
            )
        except ImportError:
            raise ImportError("aioboto3 not installed")
    
    async def upload(self, local_path: Path, remote_path: str) -> bool:
        """Upload to S3."""
        try:
            async with self.session.client('s3', region_name=self.region) as s3:
                if local_path.is_file():
                    return await self._upload_file_s3(s3, local_path, remote_path)
                else:
                    return await self._upload_directory_s3(s3, local_path, remote_path)
        except Exception:
            self.logger.exception(
                f"Failed to upload to S3: {remote_path}",
                source_module=self._source_module
            )
            return False
    
    async def _upload_file_s3(self, s3: Any, local_path: Path, remote_path: str) -> bool:
        """Upload single file to S3."""
        try:
            # Calculate checksum
            local_checksum = await self._calculate_checksum(local_path)
            
            # Upload file
            async with aiofiles.open(local_path, 'rb') as f:
                content = await f.read()
            
            await s3.put_object(
                Bucket=self.bucket_name,
                Key=remote_path,
                Body=content,
                Metadata={
                    'checksum': local_checksum,
                    'original_path': str(local_path)
                }
            )
            
            # Verify upload
            response = await s3.head_object(Bucket=self.bucket_name, Key=remote_path)
            remote_checksum = response.get('Metadata', {}).get('checksum')
            
            if remote_checksum != local_checksum:
                self.logger.error(
                    f"Checksum mismatch after upload: {remote_path}",
                    source_module=self._source_module
                )
                return False
            
            self.logger.info(
                f"Successfully uploaded file to S3: {remote_path}",
                source_module=self._source_module,
                context={
                    "size_bytes": local_path.stat().st_size,
                    "checksum": local_checksum
                }
            )
            return True
            
        except Exception:
            self.logger.exception(
                f"Failed to upload file to S3: {remote_path}",
                source_module=self._source_module
            )
            return False
    
    async def _upload_directory_s3(self, s3: Any, local_path: Path, remote_path: str) -> bool:
        """Upload directory recursively to S3."""
        try:
            upload_tasks = []
            
            for file_path in local_path.rglob("*"):
                if file_path.is_file():
                    relative_path = file_path.relative_to(local_path)
                    remote_file_path = f"{remote_path}/{relative_path}".replace("\\", "/")
                    
                    task = self._upload_file_s3(s3, file_path, remote_file_path)
                    upload_tasks.append(task)
            
            results = await asyncio.gather(*upload_tasks, return_exceptions=True)
            
            # Check results
            failed_count = sum(1 for r in results if isinstance(r, Exception) or not r)
            if failed_count > 0:
                self.logger.error(
                    f"Failed to upload {failed_count}/{len(results)} files",
                    source_module=self._source_module
                )
                return False
            
            self.logger.info(
                f"Successfully uploaded directory to S3: {remote_path}",
                source_module=self._source_module,
                context={"file_count": len(results)}
            )
            return True
            
        except Exception:
            self.logger.exception(
                f"Failed to upload directory to S3: {remote_path}",
                source_module=self._source_module
            )
            return False
    
    async def download(self, remote_path: str, local_path: Path) -> bool:
        """Download from S3."""
        try:
            async with self.session.client('s3', region_name=self.region) as s3:
                # List all objects with prefix
                paginator = s3.get_paginator('list_objects_v2')
                pages = paginator.paginate(Bucket=self.bucket_name, Prefix=remote_path)
                
                download_tasks = []
                async for page in pages:
                    for obj in page.get('Contents', []):
                        # Calculate local file path
                        relative_path = obj['Key'][len(remote_path):].lstrip("/")
                        if not relative_path:  # Single file
                            file_path = local_path
                        else:
                            file_path = local_path / relative_path
                        
                        task = self._download_file_s3(s3, obj['Key'], file_path)
                        download_tasks.append(task)
                
                if not download_tasks:
                    self.logger.error(
                        f"No files found in S3: {remote_path}",
                        source_module=self._source_module
                    )
                    return False
                
                results = await asyncio.gather(*download_tasks, return_exceptions=True)
                
                # Check results
                failed_count = sum(1 for r in results if isinstance(r, Exception) or not r)
                if failed_count > 0:
                    self.logger.error(
                        f"Failed to download {failed_count}/{len(results)} files",
                        source_module=self._source_module
                    )
                    return False
                
                return True
                
        except Exception:
            self.logger.exception(
                f"Failed to download from S3: {remote_path}",
                source_module=self._source_module
            )
            return False
    
    async def _download_file_s3(self, s3: Any, key: str, local_path: Path) -> bool:
        """Download single file from S3."""
        try:
            # Create parent directory
            local_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Download file
            response = await s3.get_object(Bucket=self.bucket_name, Key=key)
            content = await response['Body'].read()
            
            # Write async
            async with aiofiles.open(local_path, 'wb') as f:
                await f.write(content)
            
            # Verify checksum if available
            metadata = response.get('Metadata', {})
            if 'checksum' in metadata:
                local_checksum = await self._calculate_checksum(local_path)
                if local_checksum != metadata['checksum']:
                    self.logger.error(
                        f"Checksum mismatch after download: {local_path}",
                        source_module=self._source_module
                    )
                    return False
            
            return True
            
        except Exception:
            self.logger.exception(
                f"Failed to download file from S3: {key}",
                source_module=self._source_module
            )
            return False
    
    async def delete(self, remote_path: str) -> bool:
        """Delete from S3."""
        try:
            async with self.session.client('s3', region_name=self.region) as s3:
                # List all objects with prefix
                paginator = s3.get_paginator('list_objects_v2')
                pages = paginator.paginate(Bucket=self.bucket_name, Prefix=remote_path)
                
                objects_to_delete = []
                async for page in pages:
                    for obj in page.get('Contents', []):
                        objects_to_delete.append({'Key': obj['Key']})
                
                if objects_to_delete:
                    # Delete in batches of 1000 (S3 limit)
                    for i in range(0, len(objects_to_delete), 1000):
                        batch = objects_to_delete[i:i+1000]
                        await s3.delete_objects(
                            Bucket=self.bucket_name,
                            Delete={'Objects': batch}
                        )
                
                self.logger.info(
                    f"Deleted {len(objects_to_delete)} files from S3: {remote_path}",
                    source_module=self._source_module
                )
                return True
                
        except Exception:
            self.logger.exception(
                f"Failed to delete from S3: {remote_path}",
                source_module=self._source_module
            )
            return False
    
    async def exists(self, remote_path: str) -> bool:
        """Check if exists in S3."""
        try:
            async with self.session.client('s3', region_name=self.region) as s3:
                try:
                    await s3.head_object(Bucket=self.bucket_name, Key=remote_path)
                    return True
                except s3.exceptions.NoSuchKey:
                    return False
        except Exception:
            return False
    
    async def _calculate_checksum(self, file_path: Path) -> str:
        """Calculate SHA256 checksum of file."""
        sha256_hash = hashlib.sha256()
        
        async with aiofiles.open(file_path, "rb") as f:
            while chunk := await f.read(8192):
                sha256_hash.update(chunk)
        
        return sha256_hash.hexdigest() 