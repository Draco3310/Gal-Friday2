# Task: Finalize type hints and replace placeholder handler types with concrete classes

### 1. Context
- **File:** `gal_friday/logger_service.py`
- **Line:** `164-168`
- **Keyword/Pattern:** `"Placeholder"`
- **Current State:** Placeholder handler types that need to be replaced with concrete logging handler implementations

### 2. Problem Statement
The logger service contains placeholder handler types with incomplete type hints, preventing proper type checking and limiting the logging system's functionality. This creates maintenance issues and reduces code reliability in production environments.

### 3. Proposed Solution (Enterprise-Grade)

#### a. High-Level Plan
1. **Define Concrete Handler Classes:** Implement all logging handler types with proper functionality
2. **Complete Type Annotations:** Add comprehensive type hints for all logger components
3. **Implement Handler Factory:** Factory pattern for creating appropriate handler instances
4. **Add Handler Configuration:** Configurable handler settings and parameters
5. **Create Handler Registry:** Registration system for custom handlers
6. **Build Handler Monitoring:** Performance tracking and health monitoring for handlers

#### b. Pseudocode or Implementation Sketch
```python
from typing import Dict, List, Optional, Any, Union, Type, Protocol, TextIO
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
import logging
import logging.handlers
import json
import time
from pathlib import Path
from enum import Enum

class LogLevel(str, Enum):
    """Log levels"""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"

class HandlerType(str, Enum):
    """Types of logging handlers"""
    CONSOLE = "console"
    FILE = "file"
    ROTATING_FILE = "rotating_file"
    TIMED_ROTATING_FILE = "timed_rotating_file"
    SYSLOG = "syslog"
    HTTP = "http"
    SMTP = "smtp"
    ELASTICSEARCH = "elasticsearch"
    INFLUXDB = "influxdb"
    KAFKA = "kafka"
    CUSTOM = "custom"

@dataclass
class HandlerConfig:
    """Configuration for logging handlers"""
    handler_type: HandlerType
    name: str
    level: LogLevel = LogLevel.INFO
    format_string: Optional[str] = None
    parameters: Dict[str, Any] = field(default_factory=dict)
    enabled: bool = True
    filters: List[str] = field(default_factory=list)

class LogHandlerProtocol(Protocol):
    """Protocol for logging handlers"""
    
    def emit(self, record: logging.LogRecord) -> None:
        """Emit a log record"""
        ...
    
    def close(self) -> None:
        """Close the handler"""
        ...
    
    def flush(self) -> None:
        """Flush any pending output"""
        ...

class BaseLogHandler(logging.Handler, ABC):
    """Base class for custom logging handlers"""
    
    def __init__(self, config: HandlerConfig):
        super().__init__()
        self.config = config
        self.setLevel(getattr(logging, config.level.value))
        
        # Performance tracking
        self.emit_count = 0
        self.error_count = 0
        self.last_emit_time = 0.0
        
        # Setup formatter
        if config.format_string:
            formatter = logging.Formatter(config.format_string)
            self.setFormatter(formatter)
    
    @abstractmethod
    def emit(self, record: logging.LogRecord) -> None:
        """Emit a log record - must be implemented by subclasses"""
        pass
    
    def handle_emit_error(self, record: logging.LogRecord, error: Exception) -> None:
        """Handle errors during record emission"""
        self.error_count += 1
        self.handleError(record)

class ConsoleLogHandler(BaseLogHandler):
    """Concrete console logging handler"""
    
    def __init__(self, config: HandlerConfig):
        super().__init__(config)
        self.stream = self.config.parameters.get('stream', 'stdout')
        
        if self.stream == 'stderr':
            import sys
            self.target_stream = sys.stderr
        else:
            import sys
            self.target_stream = sys.stdout
    
    def emit(self, record: logging.LogRecord) -> None:
        """Emit record to console"""
        try:
            self.emit_count += 1
            self.last_emit_time = time.time()
            
            msg = self.format(record)
            
            # Add color coding if enabled
            if self.config.parameters.get('colorize', False):
                msg = self._colorize_message(msg, record.levelno)
            
            self.target_stream.write(msg + '\n')
            self.target_stream.flush()
            
        except Exception as e:
            self.handle_emit_error(record, e)
    
    def _colorize_message(self, message: str, level: int) -> str:
        """Add ANSI color codes based on log level"""
        color_map = {
            logging.DEBUG: '\033[36m',    # Cyan
            logging.INFO: '\033[32m',     # Green
            logging.WARNING: '\033[33m',  # Yellow
            logging.ERROR: '\033[31m',    # Red
            logging.CRITICAL: '\033[35m'  # Magenta
        }
        reset = '\033[0m'
        
        color = color_map.get(level, '')
        return f"{color}{message}{reset}"

class RotatingFileLogHandler(BaseLogHandler):
    """Concrete rotating file logging handler"""
    
    def __init__(self, config: HandlerConfig):
        super().__init__(config)
        
        # Extract file rotation parameters
        self.filename = config.parameters.get('filename', 'app.log')
        self.max_bytes = config.parameters.get('max_bytes', 10 * 1024 * 1024)  # 10MB
        self.backup_count = config.parameters.get('backup_count', 5)
        
        # Create the actual rotating file handler
        self.file_handler = logging.handlers.RotatingFileHandler(
            filename=self.filename,
            maxBytes=self.max_bytes,
            backupCount=self.backup_count
        )
        
        if self.formatter:
            self.file_handler.setFormatter(self.formatter)
    
    def emit(self, record: logging.LogRecord) -> None:
        """Emit record to rotating file"""
        try:
            self.emit_count += 1
            self.last_emit_time = time.time()
            
            self.file_handler.emit(record)
            
        except Exception as e:
            self.handle_emit_error(record, e)
    
    def close(self) -> None:
        """Close the file handler"""
        self.file_handler.close()
        super().close()

class ElasticsearchLogHandler(BaseLogHandler):
    """Concrete Elasticsearch logging handler"""
    
    def __init__(self, config: HandlerConfig):
        super().__init__(config)
        
        # Elasticsearch configuration
        self.hosts = config.parameters.get('hosts', ['localhost:9200'])
        self.index_pattern = config.parameters.get('index_pattern', 'logs-%Y.%m.%d')
        self.doc_type = config.parameters.get('doc_type', '_doc')
        
        # Initialize Elasticsearch client (placeholder for actual implementation)
        self.es_client = self._create_es_client()
        
        # Batch processing
        self.batch_size = config.parameters.get('batch_size', 100)
        self.batch_timeout = config.parameters.get('batch_timeout', 5.0)
        self.batch_buffer: List[Dict[str, Any]] = []
        self.last_flush_time = time.time()
    
    def _create_es_client(self):
        """Create Elasticsearch client"""
        # Placeholder for actual Elasticsearch client creation
        # from elasticsearch import Elasticsearch
        # return Elasticsearch(self.hosts)
        return None
    
    def emit(self, record: logging.LogRecord) -> None:
        """Emit record to Elasticsearch"""
        try:
            self.emit_count += 1
            self.last_emit_time = time.time()
            
            # Convert log record to document
            doc = self._record_to_document(record)
            
            # Add to batch buffer
            self.batch_buffer.append(doc)
            
            # Check if batch should be flushed
            if (len(self.batch_buffer) >= self.batch_size or 
                time.time() - self.last_flush_time > self.batch_timeout):
                self._flush_batch()
            
        except Exception as e:
            self.handle_emit_error(record, e)
    
    def _record_to_document(self, record: logging.LogRecord) -> Dict[str, Any]:
        """Convert log record to Elasticsearch document"""
        
        doc = {
            '@timestamp': time.strftime('%Y-%m-%dT%H:%M:%S.%fZ', time.gmtime(record.created)),
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno,
            'thread': record.thread,
            'process': record.process
        }
        
        # Add exception info if present
        if record.exc_info:
            doc['exception'] = self.format(record)
        
        # Add extra fields
        if hasattr(record, 'extra_fields'):
            doc.update(record.extra_fields)
        
        return doc
    
    def _flush_batch(self) -> None:
        """Flush batch buffer to Elasticsearch"""
        
        if not self.batch_buffer or not self.es_client:
            return
        
        try:
            # Bulk index operation (placeholder)
            # self.es_client.bulk(body=self._create_bulk_body())
            
            self.batch_buffer.clear()
            self.last_flush_time = time.time()
            
        except Exception as e:
            self.error_count += 1
            # Log to fallback handler or stderr

class InfluxDBLogHandler(BaseLogHandler):
    """Concrete InfluxDB logging handler for time-series log data"""
    
    def __init__(self, config: HandlerConfig):
        super().__init__(config)
        
        # InfluxDB configuration
        self.host = config.parameters.get('host', 'localhost')
        self.port = config.parameters.get('port', 8086)
        self.database = config.parameters.get('database', 'logs')
        self.measurement = config.parameters.get('measurement', 'application_logs')
        
        # Initialize InfluxDB client (placeholder)
        self.influx_client = self._create_influx_client()
        
        # Batch processing
        self.batch_size = config.parameters.get('batch_size', 100)
        self.batch_buffer: List[Dict[str, Any]] = []
    
    def _create_influx_client(self):
        """Create InfluxDB client"""
        # Placeholder for actual InfluxDB client creation
        # from influxdb import InfluxDBClient
        # return InfluxDBClient(host=self.host, port=self.port, database=self.database)
        return None
    
    def emit(self, record: logging.LogRecord) -> None:
        """Emit record to InfluxDB"""
        try:
            self.emit_count += 1
            self.last_emit_time = time.time()
            
            # Convert log record to InfluxDB point
            point = self._record_to_point(record)
            
            # Add to batch buffer
            self.batch_buffer.append(point)
            
            # Flush if batch is full
            if len(self.batch_buffer) >= self.batch_size:
                self._flush_batch()
            
        except Exception as e:
            self.handle_emit_error(record, e)
    
    def _record_to_point(self, record: logging.LogRecord) -> Dict[str, Any]:
        """Convert log record to InfluxDB point"""
        
        return {
            'measurement': self.measurement,
            'time': int(record.created * 1000000000),  # Nanoseconds
            'tags': {
                'level': record.levelname,
                'logger': record.name,
                'module': record.module,
                'function': record.funcName
            },
            'fields': {
                'message': record.getMessage(),
                'line': record.lineno,
                'thread': record.thread,
                'process': record.process
            }
        }

class LogHandlerFactory:
    """Factory for creating logging handlers with proper type annotations"""
    
    # Mapping of handler types to concrete classes
    HANDLER_CLASSES: Dict[HandlerType, Type[BaseLogHandler]] = {
        HandlerType.CONSOLE: ConsoleLogHandler,
        HandlerType.ROTATING_FILE: RotatingFileLogHandler,
        HandlerType.ELASTICSEARCH: ElasticsearchLogHandler,
        HandlerType.INFLUXDB: InfluxDBLogHandler,
    }
    
    @classmethod
    def create_handler(cls, config: HandlerConfig) -> BaseLogHandler:
        """
        Create logging handler from configuration
        Replace placeholder handler types with concrete implementations
        """
        
        handler_class = cls.HANDLER_CLASSES.get(config.handler_type)
        
        if not handler_class:
            raise ValueError(f"Unsupported handler type: {config.handler_type}")
        
        try:
            handler = handler_class(config)
            
            # Apply filters if configured
            for filter_name in config.filters:
                log_filter = cls._create_filter(filter_name)
                if log_filter:
                    handler.addFilter(log_filter)
            
            return handler
            
        except Exception as e:
            raise RuntimeError(f"Failed to create handler {config.name}: {e}")
    
    @classmethod
    def register_handler_class(cls, handler_type: HandlerType, handler_class: Type[BaseLogHandler]) -> None:
        """Register custom handler class"""
        cls.HANDLER_CLASSES[handler_type] = handler_class
    
    @classmethod
    def _create_filter(cls, filter_name: str) -> Optional[logging.Filter]:
        """Create logging filter by name"""
        # Placeholder for filter creation logic
        return None

class EnterpriseLoggerService:
    """Enterprise logging service with concrete handler implementations"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Handler registry with proper type annotations
        self.handlers: Dict[str, BaseLogHandler] = {}
        self.handler_configs: Dict[str, HandlerConfig] = {}
        
        # Performance tracking
        self.handler_stats: Dict[str, Dict[str, Any]] = {}
        
        # Initialize handlers from configuration
        self._initialize_handlers()
    
    def _initialize_handlers(self) -> None:
        """Initialize logging handlers from configuration"""
        
        handler_configs = self.config.get('handlers', [])
        
        for handler_config_data in handler_configs:
            try:
                # Create handler configuration
                config = HandlerConfig(
                    handler_type=HandlerType(handler_config_data['type']),
                    name=handler_config_data['name'],
                    level=LogLevel(handler_config_data.get('level', 'INFO')),
                    format_string=handler_config_data.get('format'),
                    parameters=handler_config_data.get('parameters', {}),
                    enabled=handler_config_data.get('enabled', True),
                    filters=handler_config_data.get('filters', [])
                )
                
                # Create and register handler
                if config.enabled:
                    handler = LogHandlerFactory.create_handler(config)
                    self.handlers[config.name] = handler
                    self.handler_configs[config.name] = config
                    
                    self.logger.info(f"Initialized handler: {config.name} ({config.handler_type.value})")
                
            except Exception as e:
                self.logger.error(f"Failed to initialize handler: {e}")
    
    def get_handler(self, name: str) -> Optional[BaseLogHandler]:
        """Get handler by name with proper type annotation"""
        return self.handlers.get(name)
    
    def add_handler(self, config: HandlerConfig) -> bool:
        """Add new handler at runtime"""
        
        try:
            if config.name in self.handlers:
                self.logger.warning(f"Handler {config.name} already exists")
                return False
            
            handler = LogHandlerFactory.create_handler(config)
            self.handlers[config.name] = handler
            self.handler_configs[config.name] = config
            
            self.logger.info(f"Added handler: {config.name}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to add handler {config.name}: {e}")
            return False
    
    def remove_handler(self, name: str) -> bool:
        """Remove handler by name"""
        
        if name in self.handlers:
            handler = self.handlers[name]
            handler.close()
            
            del self.handlers[name]
            del self.handler_configs[name]
            
            self.logger.info(f"Removed handler: {name}")
            return True
        
        return False
    
    def get_handler_statistics(self) -> Dict[str, Dict[str, Any]]:
        """Get statistics for all handlers"""
        
        stats = {}
        
        for name, handler in self.handlers.items():
            stats[name] = {
                'type': self.handler_configs[name].handler_type.value,
                'emit_count': handler.emit_count,
                'error_count': handler.error_count,
                'last_emit_time': handler.last_emit_time,
                'enabled': self.handler_configs[name].enabled
            }
        
        return stats
    
    def flush_all_handlers(self) -> None:
        """Flush all handlers"""
        
        for handler in self.handlers.values():
            try:
                handler.flush()
            except Exception as e:
                self.logger.error(f"Error flushing handler: {e}")
    
    def close_all_handlers(self) -> None:
        """Close all handlers"""
        
        for handler in self.handlers.values():
            try:
                handler.close()
            except Exception as e:
                self.logger.error(f"Error closing handler: {e}")
```

#### c. Key Considerations & Dependencies
- **Type Safety:** Complete type annotations for all handler classes and factory methods
- **Performance:** Efficient handler implementations with batching and asynchronous operations
- **Configuration:** Flexible handler configuration with validation and runtime updates
- **Monitoring:** Handler performance tracking and error reporting

### 4. Acceptance Criteria
- [ ] All placeholder handler types replaced with concrete implementations
- [ ] Complete type annotations for all logging components
- [ ] Handler factory pattern with proper type safety
- [ ] Support for multiple handler types (console, file, Elasticsearch, InfluxDB)
- [ ] Configurable handler parameters and filtering
- [ ] Handler registration system for custom implementations
- [ ] Performance monitoring and statistics for all handlers
- [ ] Proper error handling and fallback mechanisms
- [ ] Runtime handler management (add, remove, configure)
- [ ] Comprehensive testing for all handler types
- [ ] Placeholder handler types are completely replaced with production implementations 