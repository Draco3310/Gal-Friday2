"""Performance tests for Gal-Friday system."""

from datetime import UTC, datetime
from decimal import Decimal
import random
import statistics
import time
from typing import Any

import asyncio
import numpy as np
import psutil
from rich import print as rich_print

from gal_friday.core.events import (
    EventType,
    MarketDataEvent,
    PredictionEvent,
    TradeSignalProposedEvent,
)
from gal_friday.core.pubsub import PubSubManager


class PerformanceTestRunner:
    """Run performance benchmarks on Gal-Friday components."""

    def __init__(self):
        self.results = {}
        self.pubsub = PubSubManager()

    async def test_event_throughput(self, duration_seconds: int = 10):
        """Test event publishing throughput."""
        rich_print(f"\n=== Testing Event Throughput (duration: {duration_seconds}s) ===")

        events_published = 0
        start_time = time.time()

        while time.time() - start_time < duration_seconds:
            event = MarketDataEvent(
                event_type=EventType.PRICE_UPDATE,
                timestamp=datetime.now(UTC),
                trading_pair="XRP/USD",
                price=Decimal(str(random.uniform(0.4, 0.6))),
                volume=Decimal(str(random.uniform(100000, 1000000))),
            )

            await self.pubsub.publish(EventType.PRICE_UPDATE, event)
            events_published += 1

        elapsed = time.time() - start_time
        throughput = events_published / elapsed

        self.results["event_throughput"] = {
            "events_published": events_published,
            "duration_seconds": elapsed,
            "events_per_second": throughput,
        }

        rich_print(f"Published {events_published:,} events in {elapsed:.2f}s")
        rich_print(f"Throughput: {throughput:,.0f} events/second")

    async def test_prediction_latency(self, num_predictions: int = 1000):
        """Test prediction generation latency."""
        rich_print(f"\n=== Testing Prediction Latency (n={num_predictions}) ===")

        latencies = []

        for _ in range(num_predictions):
            features = {
                f"feature_{i}": random.random()
                for i in range(50)  # 50 features
            }

            start = time.perf_counter()

            # Simulate prediction processing
            prediction_value = sum(features.values()) / len(features)

            # Create prediction event
            event = PredictionEvent(
                event_type=EventType.PREDICTION_GENERATED,
                timestamp=datetime.now(UTC),
                model_id="test_model",
                trading_pair="XRP/USD",
                prediction_value=prediction_value,
                confidence=random.uniform(0.7, 0.95),
                associated_features=features,
            )

            await self.pubsub.publish(EventType.PREDICTION_GENERATED, event)

            latency = (time.perf_counter() - start) * 1000  # ms
            latencies.append(latency)

        self.results["prediction_latency"] = {
            "count": num_predictions,
            "mean_ms": statistics.mean(latencies),
            "median_ms": statistics.median(latencies),
            "p95_ms": np.percentile(latencies, 95),
            "p99_ms": np.percentile(latencies, 99),
            "min_ms": min(latencies),
            "max_ms": max(latencies),
        }

        rich_print(f"Latency - Mean: {self.results['prediction_latency']['mean_ms']:.2f}ms")
        rich_print(f"Latency - P95: {self.results['prediction_latency']['p95_ms']:.2f}ms")
        rich_print(f"Latency - P99: {self.results['prediction_latency']['p99_ms']:.2f}ms")

    async def test_concurrent_load(self, num_workers: int = 10, duration_seconds: int = 30):
        """Test system under concurrent load."""
        rich_print(f"\n=== Testing Concurrent Load (workers={num_workers}, duration={duration_seconds}s) ===")

        async def worker(worker_id: int):
            events_processed = 0
            errors = 0
            start_time = time.time()

            while time.time() - start_time < duration_seconds:
                try:
                    # Simulate various operations
                    operation = random.choice(["market_data", "prediction", "signal"])

                    if operation == "market_data":
                        event = MarketDataEvent(
                            event_type=EventType.PRICE_UPDATE,
                            timestamp=datetime.now(UTC),
                            trading_pair=random.choice(["XRP/USD", "DOGE/USD"]),
                            price=Decimal(str(random.uniform(0.1, 1.0))),
                            volume=Decimal(str(random.uniform(10000, 100000))),
                        )
                        await self.pubsub.publish(EventType.PRICE_UPDATE, event)

                    elif operation == "prediction":
                        event = PredictionEvent(
                            event_type=EventType.PREDICTION_GENERATED,
                            timestamp=datetime.now(UTC),
                            model_id=f"model_{worker_id}",
                            trading_pair=random.choice(["XRP/USD", "DOGE/USD"]),
                            prediction_value=random.random(),
                            confidence=random.uniform(0.5, 1.0),
                        )
                        await self.pubsub.publish(EventType.PREDICTION_GENERATED, event)

                    else:  # signal
                        event = TradeSignalProposedEvent(
                            event_type=EventType.TRADE_SIGNAL_PROPOSED,
                            timestamp=datetime.now(UTC),
                            signal_id=f"signal_{worker_id}_{events_processed}",
                            trading_pair=random.choice(["XRP/USD", "DOGE/USD"]),
                            signal=random.choice(["BUY", "SELL", "HOLD"]),
                            entry_price=Decimal(str(random.uniform(0.1, 1.0))),
                            quantity=Decimal(str(random.uniform(100, 1000))),
                        )
                        await self.pubsub.publish(EventType.TRADE_SIGNAL_PROPOSED, event)

                    events_processed += 1

                except Exception:
                    errors += 1

                # Small delay to prevent overwhelming
                await asyncio.sleep(0.001)

            return events_processed, errors

        # Run concurrent workers
        start_time = time.time()
        tasks = [worker(i) for i in range(num_workers)]
        results = await asyncio.gather(*tasks)
        elapsed = time.time() - start_time

        total_events = sum(r[0] for r in results)
        total_errors = sum(r[1] for r in results)

        self.results["concurrent_load"] = {
            "workers": num_workers,
            "duration_seconds": elapsed,
            "total_events": total_events,
            "total_errors": total_errors,
            "events_per_second": total_events / elapsed,
            "error_rate": total_errors / total_events if total_events > 0 else 0,
        }

        rich_print(f"Processed {total_events:,} events with {num_workers} workers")
        rich_print(f"Throughput: {total_events / elapsed:,.0f} events/second")
        rich_print(f"Error rate: {self.results['concurrent_load']['error_rate']:.2%}")

    async def test_memory_usage(self, duration_seconds: int = 60):
        """Test memory usage under load."""
        rich_print(f"\n=== Testing Memory Usage (duration={duration_seconds}s) ===")

        process = psutil.Process()
        memory_samples = []

        start_time = time.time()

        # Generate load while monitoring memory
        async def generate_load():
            while time.time() - start_time < duration_seconds:
                # Create various objects to stress memory
                {
                    "features": np.random.rand(1000, 50),  # 1000 samples, 50 features
                    "predictions": [random.random() for _ in range(1000)],
                    "events": [
                        MarketDataEvent(
                            event_type=EventType.PRICE_UPDATE,
                            timestamp=datetime.now(UTC),
                            trading_pair="XRP/USD",
                            price=Decimal(str(random.random())),
                            volume=Decimal(str(random.random() * 1000000)),
                        ) for _ in range(100)
                    ],
                }

                await asyncio.sleep(0.1)

        # Monitor memory in parallel
        async def monitor_memory():
            while time.time() - start_time < duration_seconds:
                memory_info = process.memory_info()
                memory_samples.append({
                    "timestamp": time.time() - start_time,
                    "rss_mb": memory_info.rss / 1024 / 1024,
                    "vms_mb": memory_info.vms / 1024 / 1024,
                })
                await asyncio.sleep(1)

        await asyncio.gather(generate_load(), monitor_memory())

        rss_values = [s["rss_mb"] for s in memory_samples]

        self.results["memory_usage"] = {
            "samples": len(memory_samples),
            "initial_rss_mb": memory_samples[0]["rss_mb"] if memory_samples else 0,
            "peak_rss_mb": max(rss_values) if rss_values else 0,
            "final_rss_mb": memory_samples[-1]["rss_mb"] if memory_samples else 0,
            "mean_rss_mb": statistics.mean(rss_values) if rss_values else 0,
            "memory_growth_mb": (memory_samples[-1]["rss_mb"] - memory_samples[0]["rss_mb"]) if memory_samples else 0,
        }

        rich_print(f"Initial memory: {self.results['memory_usage']['initial_rss_mb']:.1f}MB")
        rich_print(f"Peak memory: {self.results['memory_usage']['peak_rss_mb']:.1f}MB")
        rich_print(f"Memory growth: {self.results['memory_usage']['memory_growth_mb']:.1f}MB")

    async def test_cache_performance(self):
        """Test cache hit rates and performance."""
        rich_print("\n=== Testing Cache Performance ===")

        from gal_friday.utils.performance_optimizer import LRUCache

        cache = LRUCache[dict[str, Any]](maxsize=1000)

        # Warm up cache
        for i in range(500):
            await cache.set(f"key_{i}", {"data": f"value_{i}"})

        # Test cache performance
        operations = 10000
        cache_times = []

        for i in range(operations):
            key = f"key_{random.randint(0, 999)}"

            start = time.perf_counter()
            result = await cache.get(key)
            if result is None:
                await cache.set(key, {"data": f"value_{i}"})
            elapsed = time.perf_counter() - start

            cache_times.append(elapsed * 1000000)  # microseconds

        stats = cache.get_stats()

        self.results["cache_performance"] = {
            "operations": operations,
            "hit_rate": stats["hit_rate"],
            "mean_latency_us": statistics.mean(cache_times),
            "p95_latency_us": np.percentile(cache_times, 95),
            "p99_latency_us": np.percentile(cache_times, 99),
        }

        rich_print(f"Cache hit rate: {stats['hit_rate']:.1%}")
        rich_print(f"Mean latency: {self.results['cache_performance']['mean_latency_us']:.1f}μs")
        rich_print(f"P99 latency: {self.results['cache_performance']['p99_latency_us']:.1f}μs")

    async def test_database_connection_pool(self):
        """Test database connection pool performance."""
        rich_print("\n=== Testing Connection Pool Performance ===")

        from gal_friday.utils.performance_optimizer import ConnectionPool

        # Mock connection creation
        async def create_mock_connection():
            await asyncio.sleep(0.01)  # Simulate connection time
            return {"id": random.randint(1000, 9999), "created": time.time()}

        pool = ConnectionPool(
            create_conn=create_mock_connection,
            max_connections=10,
            min_connections=2,
        )

        await pool.start()

        # Test concurrent connection usage
        async def use_connection(duration: float):
            conn = await pool.acquire()
            await asyncio.sleep(duration)
            await pool.release(conn)

        # Run concurrent operations
        num_operations = 100
        operation_times = []

        for _ in range(num_operations):
            start = time.perf_counter()
            await use_connection(random.uniform(0.01, 0.05))
            elapsed = time.perf_counter() - start
            operation_times.append(elapsed * 1000)  # ms

        pool_stats = pool.get_stats()

        self.results["connection_pool"] = {
            "operations": num_operations,
            "mean_time_ms": statistics.mean(operation_times),
            "p95_time_ms": np.percentile(operation_times, 95),
            "pool_stats": pool_stats,
        }

        rich_print(f"Mean operation time: {self.results['connection_pool']['mean_time_ms']:.1f}ms")
        rich_print(f"Pool stats: {pool_stats}")

        await pool.stop()

    def generate_report(self) -> str:
        """Generate performance test report."""
        report = ["=" * 70]
        report.append("GAL-FRIDAY PERFORMANCE TEST REPORT")
        report.append("=" * 70)
        report.append(f"Generated: {datetime.now(UTC).isoformat()}")
        report.append("")

        # Event Throughput
        if "event_throughput" in self.results:
            report.append("EVENT THROUGHPUT")
            report.append("-" * 30)
            report.append(f"Events/second: {self.results['event_throughput']['events_per_second']:,.0f}")
            report.append("")

        # Prediction Latency
        if "prediction_latency" in self.results:
            report.append("PREDICTION LATENCY")
            report.append("-" * 30)
            report.append(f"Mean: {self.results['prediction_latency']['mean_ms']:.2f}ms")
            report.append(f"P95: {self.results['prediction_latency']['p95_ms']:.2f}ms")
            report.append(f"P99: {self.results['prediction_latency']['p99_ms']:.2f}ms")
            report.append("")

        # Concurrent Load
        if "concurrent_load" in self.results:
            report.append("CONCURRENT LOAD TEST")
            report.append("-" * 30)
            report.append(f"Workers: {self.results['concurrent_load']['workers']}")
            report.append(f"Total events: {self.results['concurrent_load']['total_events']:,}")
            report.append(f"Events/second: {self.results['concurrent_load']['events_per_second']:,.0f}")
            report.append(f"Error rate: {self.results['concurrent_load']['error_rate']:.2%}")
            report.append("")

        # Memory Usage
        if "memory_usage" in self.results:
            report.append("MEMORY USAGE")
            report.append("-" * 30)
            report.append(f"Initial: {self.results['memory_usage']['initial_rss_mb']:.1f}MB")
            report.append(f"Peak: {self.results['memory_usage']['peak_rss_mb']:.1f}MB")
            report.append(f"Growth: {self.results['memory_usage']['memory_growth_mb']:.1f}MB")
            report.append("")

        # Cache Performance
        if "cache_performance" in self.results:
            report.append("CACHE PERFORMANCE")
            report.append("-" * 30)
            report.append(f"Hit rate: {self.results['cache_performance']['hit_rate']:.1%}")
            report.append(f"Mean latency: {self.results['cache_performance']['mean_latency_us']:.1f}μs")
            report.append("")

        # Performance Targets
        report.append("PERFORMANCE TARGETS")
        report.append("-" * 30)

        targets_met = []
        targets_missed = []

        # Check against targets
        if "event_throughput" in self.results:
            if self.results["event_throughput"]["events_per_second"] >= 10000:
                targets_met.append("✓ Event throughput > 10,000/s")
            else:
                targets_missed.append("✗ Event throughput < 10,000/s")

        if "prediction_latency" in self.results:
            if self.results["prediction_latency"]["p99_ms"] <= 50:
                targets_met.append("✓ Prediction P99 latency < 50ms")
            else:
                targets_missed.append("✗ Prediction P99 latency > 50ms")

        if "memory_usage" in self.results:
            if self.results["memory_usage"]["memory_growth_mb"] <= 100:
                targets_met.append("✓ Memory growth < 100MB")
            else:
                targets_missed.append("✗ Memory growth > 100MB")

        for target in targets_met:
            report.append(target)
        for target in targets_missed:
            report.append(target)

        report.append("")
        report.append("=" * 70)

        return "\n".join(report)


async def run_all_performance_tests():
    """Run all performance tests."""
    runner = PerformanceTestRunner()

    rich_print("Starting Gal-Friday Performance Tests...")
    rich_print("This will take several minutes to complete.")

    # Run tests
    await runner.test_event_throughput(duration_seconds=10)
    await runner.test_prediction_latency(num_predictions=1000)
    await runner.test_concurrent_load(num_workers=10, duration_seconds=30)
    await runner.test_memory_usage(duration_seconds=30)
    await runner.test_cache_performance()
    await runner.test_database_connection_pool()

    # Generate report
    report = runner.generate_report()
    rich_print("\n" + report)

    # Save report
    with open("performance_test_report.txt", "w") as f:
        f.write(report)
    rich_print("\nReport saved to: performance_test_report.txt")

    return runner.results


if __name__ == "__main__":
    asyncio.run(run_all_performance_tests())
