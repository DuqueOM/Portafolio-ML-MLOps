-- Example PostGIS schema for trips table
CREATE TABLE IF NOT EXISTS trips (
  id SERIAL PRIMARY KEY,
  start_ts TIMESTAMP NOT NULL,
  duration_seconds INTEGER NOT NULL,
  weather_conditions VARCHAR(16) NOT NULL,
  geom GEOGRAPHY(POINT, 4326)
);

-- Spatial index
CREATE INDEX IF NOT EXISTS idx_trips_geom ON trips USING GIST(geom);
