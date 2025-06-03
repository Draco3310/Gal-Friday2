# File: gal_friday/kraken_historical_data_service.py
# Method: _fetch_ohlcv_data_from_api (implementing actual API call)
# TODO: Lines 504-505 - Implement actual API call using aiohttp or ccxt

# ... (preceding code in KrakenHistoricalDataService class, including __init__, _make_public_request, _get_kraken_pair_name) ...

    async def _fetch_ohlcv_data_from_api(
        self,
        trading_pair: str,
        start_time: datetime, # Expected to be timezone-aware (UTC)
        end_time: datetime,   # Expected to be timezone-aware (UTC)
        interval_str: str,    # e.g., "1m", "5m", "1h", "1d"
    ) -> pd.DataFrame | None:
        """Fetch OHLCV data directly from Kraken API for the given range and interval."""
        self.logger.info(
            "Fetching OHLCV from API for %s: %s to %s, interval %s",
            trading_pair, start_time, end_time, interval_str,
            source_module=self._source_module
        )

        kraken_pair_name = self._get_kraken_pair_name(trading_pair)
        if not kraken_pair_name:
            self.logger.error(
                "Could not get Kraken pair name for %s", trading_pair,
                source_module=self._source_module
            )
            return None

        kraken_interval_code = self._map_interval_to_kraken_code(interval_str)
        if kraken_interval_code is None:
            self.logger.error(
                "Unsupported interval string for Kraken API: %s", interval_str,
                source_module=self._source_module
            )
            return None

        all_ohlcv_data = []
        current_since_timestamp = int(start_time.timestamp()) # Kraken API uses Unix timestamp for 'since'

        # Kraken returns a max of 720 data points per call for OHLC.
        # We need to paginate if the requested range is larger.
        # The 'last' field in the response is the timestamp of the boundary of the last record returned,
        # and can be used as 'since' for the next request.

        MAX_API_CALLS = self.config.get("kraken_api", {}).get("ohlcv_max_pagination_calls", 20) # Safety break
        api_calls_count = 0

        while api_calls_count < MAX_API_CALLS:
            api_calls_count += 1
            params = {
                "pair": kraken_pair_name,
                "interval": kraken_interval_code,
                "since": current_since_timestamp, # 'since' is exclusive of the timestamp itself in some interpretations, or inclusive.
                                                  # Kraken's OHLC 'since' parameter means "return committed OHLC data since given id"
                                                  # The 'last' value from response is used as 'since' for next page.
            }
            
            self.logger.debug(
                "Kraken API OHLC request for %s: params=%s", trading_pair, params,
                source_module=self._source_module
            )

            # Use the existing _make_public_request which handles aiohttp call
            # This method is assumed to be part of the same class.
            # The _fetch_ohlcv_data (wrapper with circuit breaker) calls this _fetch_ohlcv_data_from_api
            # So, direct call to _make_public_request here is fine.
            response_data = await self._make_public_request("/0/public/OHLC", params)

            if not response_data or response_data.get("error"):
                error_messages = response_data.get("error", ["Unknown API error"]) if response_data else ["No response from API"]
                self.logger.error(
                    "Kraken API error fetching OHLCV for %s (call %s): %s. Params: %s",
                    trading_pair, api_calls_count, error_messages, params,
                    source_module=self._source_module
                )
                # If it's the first call and it fails, we likely have no data.
                # If subsequent calls fail, we might have partial data.
                break # Exit pagination loop on error

            result = response_data.get("result", {})
            pair_data = result.get(kraken_pair_name) # Kraken nests data under the pair key

            if not pair_data:
                self.logger.info(
                    "No more OHLCV data returned from Kraken API for %s (call %s, since: %s).",
                    trading_pair, api_calls_count, current_since_timestamp,
                    source_module=self._source_module
                )
                break # No more data for this pair

            # Process the candle data
            # Each item in pair_data is: [<time>, <open>, <high>, <low>, <close>, <vwap>, <volume>, <count>]
            for candle in pair_data:
                try:
                    timestamp_unix = int(candle[0])
                    dt_object = datetime.fromtimestamp(timestamp_unix, tz=UTC)

                    # Stop if we've fetched data beyond the requested end_time
                    # Note: Kraken's 'since' might return data starting slightly before 'since'
                    # if 'since' falls within a candle. The first candle returned can have a timestamp
                    # equal to or later than 'since'.
                    if dt_object > end_time: # If candle start time is past our desired end_time
                        self.logger.debug("Fetched OHLCV data beyond requested end_time for %s. Stopping pagination.", trading_pair)
                        # Mark to break outer loop after processing this batch
                        current_since_timestamp = result.get("last", current_since_timestamp) # Update last for safety
                        if int(current_since_timestamp) <= timestamp_unix : # if last is not advancing
                             current_since_timestamp = timestamp_unix + 1 # force advance if stuck
                        # Set a flag to break the outer while loop
                        api_calls_count = MAX_API_CALLS + 1 # Force break outer loop
                        break # Break from processing candles in this batch

                    # Only add data within the original [start_time, end_time] inclusive range
                    if dt_object >= start_time:
                         all_ohlcv_data.append({
                            "timestamp": dt_object,
                            "open": Decimal(str(candle[1])),
                            "high": Decimal(str(candle[2])),
                            "low": Decimal(str(candle[3])),
                            "close": Decimal(str(candle[4])),
                            "volume": Decimal(str(candle[6])), # candle[5] is vwap, candle[7] is count
                        })
                except (IndexError, ValueError, TypeError) as e:
                    self.logger.warning(
                        "Error processing individual OHLCV candle data for %s: %s. Candle: %s",
                        trading_pair, e, candle, source_module=self._source_module
                    )
                    continue # Skip this malformed candle
            
            if api_calls_count > MAX_API_CALLS : # Check if inner loop set flag to break outer
                break

            # Update 'since' for the next iteration using the 'last' timestamp from the response
            # 'last' is the timestamp of the last candle. For the next request, 'since' should be this 'last' value.
            last_timestamp_in_response = result.get("last")
            if last_timestamp_in_response is None:
                self.logger.warning(
                    "Kraken API OHLCV response for %s missing 'last' key for pagination. Stopping.",
                    trading_pair, source_module=self._source_module
                )
                break
            
            # If the 'last' timestamp is not advancing, it means no more new data or stuck.
            if int(last_timestamp_in_response) <= current_since_timestamp and len(pair_data) < 720 : # if last is not advancing and not a full page
                self.logger.info(
                    "Kraken API 'last' timestamp (%s) did not advance from 'since' (%s) for %s and not a full page. Assuming end of data.",
                    last_timestamp_in_response, current_since_timestamp, trading_pair,
                    source_module=self._source_module
                )
                break

            current_since_timestamp = int(last_timestamp_in_response)

            # Small delay to respect potential implicit rate limits not covered by self.rate_limiter
            # (self.rate_limiter is called by the wrapper _fetch_ohlcv_data)
            await asyncio.sleep(self.config.get("kraken_api", {}).get("ohlcv_pagination_delay_s", 0.2))

        if api_calls_count >= MAX_API_CALLS:
            self.logger.warning(
                "Reached max API calls (%s) for OHLCV pagination for %s. Data might be incomplete.",
                MAX_API_CALLS, trading_pair, source_module=self._source_module
            )

        if not all_ohlcv_data:
            self.logger.info(
                "No OHLCV data points collected from API for %s in the specified range.",
                trading_pair, source_module=self._source_module
            )
            return None

        # Create DataFrame
        try:
            df = pd.DataFrame(all_ohlcv_data)
            if df.empty:
                return None
            df = df.set_index("timestamp")
            df = df.sort_index() # Ensure chronological order
            # Remove potential duplicates that might arise from pagination logic if 'since' is inclusive
            df = df[~df.index.duplicated(keep='first')]
            
            self.logger.info(
                "Successfully fetched and processed %s OHLCV data points from API for %s.",
                len(df), trading_pair, source_module=self._source_module
            )
            return df
        except Exception as e:
            self.logger.error(
                "Failed to create DataFrame from fetched OHLCV data for %s: %s",
                trading_pair, e, exc_info=True, source_module=self._source_module
            )
            return None

    def _map_interval_to_kraken_code(self, interval_str: str) -> int | None:
        """Maps human-readable interval string to Kraken API integer code."""
        # Kraken intervals: 1, 5, 15, 30, 60 (1h), 240 (4h), 1440 (1d), 10080 (1w), 21600 (15d)
        mapping = {
            "1m": 1, "1min": 1,
            "5m": 5, "5min": 5,
            "15m": 15, "15min": 15,
            "30m": 30, "30min": 30,
            "1h": 60, "60m": 60, "60min": 60,
            "4h": 240, "240m": 240, "240min": 240,
            "1d": 1440, "1day": 1440,
            "1w": 10080, "1week": 10080, "7d": 10080,
            "15d": 21600, "15day": 21600,
        }
        return mapping.get(interval_str.lower())

# ... (rest of KrakenHistoricalDataService class, including _get_missing_ranges, _store_ohlcv_data_in_influxdb, etc.) ...

