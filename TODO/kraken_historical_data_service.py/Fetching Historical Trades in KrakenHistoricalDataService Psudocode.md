# File: gal_friday/kraken_historical_data_service.py
# Method: get_historical_trades
# TODO: Line 339 - Implement fetching trade data from Kraken API

# ... (preceding code in KrakenHistoricalDataService class) ...

    async def get_historical_trades(
        self,
        trading_pair: str,
        start_time: datetime,
        end_time: datetime,
    ) -> pd.DataFrame | None:
        """Get historical trade data for a given pair and time range.
        This method first checks InfluxDB for stored data, then fetches any missing
        data from the Kraken API using the self.fetch_trades() method.
        """
        self.logger.info(
            "Getting historical trades for %s from %s to %s",
            trading_pair,
            start_time,
            end_time,
            source_module=self._source_module,
        )

        # 1. Attempt to fetch from InfluxDB first
        # Ensure start_time and end_time are timezone-aware (UTC) if not already
        start_time_utc = start_time.replace(tzinfo=UTC) if start_time.tzinfo is None else start_time
        end_time_utc = end_time.replace(tzinfo=UTC) if end_time.tzinfo is None else end_time

        db_df = await self._query_trades_data_from_influxdb(trading_pair, start_time_utc, end_time_utc)

        # 2. Check if data from DB is complete for the requested range
        if db_df is not None and not db_df.empty:
            # Ensure DataFrame index is timezone-aware (UTC) for comparison
            if db_df.index.tz is None:
                db_df.index = db_df.index.tz_localize(UTC)
            
            available_start = db_df.index.min()
            available_end = db_df.index.max()

            if available_start <= start_time_utc and available_end >= end_time_utc:
                self.logger.info(
                    "Complete trade data found in InfluxDB for %s in range %s to %s.",
                    trading_pair, start_time_utc, end_time_utc,
                    source_module=self._source_module,
                )
                # Slice to exact requested range and return
                return db_df[(db_df.index >= start_time_utc) & (db_df.index <= end_time_utc)]
        
        self.logger.info(
            "Trade data in InfluxDB for %s is incomplete or missing for range %s to %s. Will attempt API fetch.",
            trading_pair, start_time_utc, end_time_utc,
            source_module=self._source_module
        )

        # 3. Determine missing ranges (even if db_df is None or empty, this will give the full range)
        #    The existing _get_missing_ranges might need adjustment if it assumes df.index is always valid.
        #    For simplicity here, if db_df is None or empty, the full range is missing.
        #    A more robust _get_missing_ranges (which we will address for Line 818) would be ideal.
        
        # For now, let's assume we need to fetch the entire range if not fully covered by DB.
        # A more sophisticated approach would fetch only the truly missing parts.
        # The `fetch_trades` method itself handles pagination and fetching within a range.

        # Call the existing self.fetch_trades method to get data from Kraken API
        # The `fetch_trades` method (lines 869+) already handles pagination and fetching.
        # It returns a list of dicts. We need to convert this to a DataFrame.
        
        self.logger.info(
            "Fetching historical trades from Kraken API for %s from %s to %s",
            trading_pair, start_time_utc, end_time_utc,
            source_module=self._source_module,
        )
        
        # The `fetch_trades` method in the provided code seems to be designed to fetch
        # based on `since` (timestamp or trade ID) and `limit`, not strictly start_time and end_time for a range.
        # It fetches trades *from* `since` up to `limit` or until `until` is hit.
        # We might need to adapt its usage or the method itself if we want to fill specific gaps.

        # For this TODO, let's assume we call `fetch_trades` for the overall requested range if DB data is insufficient.
        # The current `fetch_trades` takes `since` (start_time) and `until` (end_time).
        api_trades_list = await self.fetch_trades(
            trading_pair=trading_pair,
            since=start_time_utc, # Pass the original requested start_time
            until=end_time_utc,   # Pass the original requested end_time
            limit=None # Fetch all available in the range, subject to Kraken limits per call (handled by pagination in fetch_trades)
        )

        api_df = None
        if api_trades_list:
            try:
                api_df = pd.DataFrame(api_trades_list)
                if not api_df.empty:
                    api_df['timestamp'] = pd.to_datetime(api_df['timestamp'], utc=True)
                    api_df = api_df.set_index('timestamp')
                    # Ensure columns match what _store_trades_data_in_influxdb expects (price, volume, side)
                    # The `fetch_trades` method already creates 'price', 'volume', 'side'.
                    
                    # Sort by timestamp as API might not guarantee order with pagination
                    api_df = api_df.sort_index()

                    self.logger.info(
                        "Fetched %s trades from API for %s.", len(api_df), trading_pair,
                        source_module=self._source_module
                    )
                    # 4. Store fetched data into InfluxDB
                    #    Need a new method `_store_trades_data_in_influxdb` similar to `_store_ohlcv_data_in_influxdb`
                    await self._store_trades_data_in_influxdb(api_df, trading_pair)
                else:
                    api_df = None # Ensure it's None if empty after processing
            except Exception as e:
                self.logger.error(
                    "Failed to process trades from API into DataFrame for %s: %s",
                    trading_pair, e, exc_info=True, source_module=self._source_module
                )
                api_df = None
        else:
            self.logger.info(
                "No new trades fetched from API for %s in range %s to %s.",
                trading_pair, start_time_utc, end_time_utc,
                source_module=self._source_module
            )

        # 5. Combine with existing InfluxDB data (if any)
        combined_df = None
        if db_df is not None and not db_df.empty and api_df is not None and not api_df.empty:
            combined_df = pd.concat([db_df, api_df])
            # Remove duplicates, keeping the ones from API fetch if overlap (or based on a chosen strategy)
            # Assuming API data is more "raw" or InfluxDB is the source of truth for what's already processed.
            # If API data is fetched for ranges already in DB, duplicates might occur.
            combined_df = combined_df[~combined_df.index.duplicated(keep='first')] # Keep first, assuming db_df was first
            combined_df = combined_df.sort_index()
        elif api_df is not None and not api_df.empty:
            combined_df = api_df
        elif db_df is not None and not db_df.empty:
            combined_df = db_df
        # Else, combined_df remains None

        if combined_df is not None and not combined_df.empty:
            # Ensure index is timezone-aware UTC before slicing
            if combined_df.index.tz is None:
                 combined_df.index = combined_df.index.tz_localize(UTC)
            # Slice to the exact requested range
            final_df = combined_df[(combined_df.index >= start_time_utc) & (combined_df.index <= end_time_utc)]
            if not final_df.empty:
                self.logger.info(
                    "Returning %s combined trades for %s.", len(final_df), trading_pair,
                    source_module=self._source_module
                )
                return final_df
            else:
                self.logger.info(
                    "Combined trades for %s resulted in an empty DataFrame for the requested range.",
                    trading_pair, source_module=self._source_module
                )
                return None
        else:
            self.logger.warning(
                "No trade data available for %s after DB query and API fetch for range %s to %s.",
                trading_pair, start_time_utc, end_time_utc,
                source_module=self._source_module,
            )
            return None

    async def _store_trades_data_in_influxdb(self, df: pd.DataFrame, trading_pair: str) -> bool:
        """Store trade data in InfluxDB."""
        if df is None or df.empty:
            return False
        
        # Ensure 'price', 'volume', 'side' columns exist
        required_cols = ['price', 'volume', 'side']
        if not all(col in df.columns for col in required_cols):
            self.logger.error(
                f"DataFrame for InfluxDB trade storage for {trading_pair} is missing one of required columns: {required_cols}. Columns present: {df.columns.tolist()}",
                source_module=self._source_module
            )
            return False

        try:
            bucket = self.config.get("influxdb", {}).get("bucket", "gal_friday")
            points = []

            for timestamp, row in df.iterrows():
                # Ensure timestamp is timezone-aware (UTC) before writing
                ts_to_write = timestamp
                if timestamp.tzinfo is None:
                    ts_to_write = timestamp.tz_localize(UTC)
                
                point = (
                    Point(self.trades_measurement) # self.trades_measurement = "market_data_trades"
                    .tag("trading_pair", trading_pair)
                    .tag("exchange", "kraken") # Assuming kraken
                    .field("price", float(row["price"]))
                    .field("volume", float(row["volume"]))
                    .field("side", str(row["side"]))
                    # Add other fields if present in DataFrame and relevant, e.g., order_type, misc
                    # .field("order_type", str(row.get("order_type", ""))) 
                    # .field("misc", str(row.get("misc", "")))
                    .time(ts_to_write) # Pass timezone-aware datetime
                )
                points.append(point)

            if points:
                self.write_api.write(bucket=bucket, record=points)
                self.logger.info(
                    "Stored %s trade points in InfluxDB for %s",
                    len(points), trading_pair,
                    source_module=self._source_module
                )
                return True
            return False
        except Exception as e:
            self.logger.exception(
                "Error storing trade data in InfluxDB for %s: %s",
                trading_pair, e,
                source_module=self._source_module
            )
            return False

# ... (rest of KrakenHistoricalDataService class, including fetch_trades, _make_public_request, etc.) ...

