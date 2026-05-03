import ffmpeg
import json
from dataclasses import dataclass
from datetime import datetime, timedelta, UTC
import gpmf
import gpxpy
from gpxpy.gpx import GPXTrackPoint
from aerofiles.igc import Reader
from utils import Bearing, get_bearing
import pandas as pd
import numpy as np

GPX_DILUTION_THRESHOLD = 20

@dataclass
class IGCRecord:
    gpx_track_point: GPXTrackPoint
    bearing: Bearing
    pressure_altitude: float

class IGCFile:

    def __init__(self, path_to_igc):

        with open(path_to_igc, 'r') as f:
            reader = Reader()
            data = reader.read(f)
        
        assert "fix_records" in data.keys()
        assert len(data['fix_records']) == 2
        assert len(data['fix_records'][0]) == 0
        assert len(data['fix_records'][1]) > 0

        data_lst = data['fix_records'][1]
        seen = set()
        seen_add = seen.add
        data_lst = [x for x in data_lst if not (x['datetime'] in seen or seen_add(x['datetime']))]

        self.records: list[IGCRecord] = []
        for i, x in enumerate(data_lst):
            gpx_pt = GPXTrackPoint(
                latitude=x['lat'],
                longitude=x['lon'],
                elevation=x['gps_alt'],
                time=x['datetime'].replace(tzinfo=UTC)
            )
            p_alt = x['pressure_alt']
            self.records.append(
                IGCRecord(gpx_pt, None, p_alt)
            )
            if i > 0:
                self.records[i-1].bearing = get_bearing(self.records[i-1].gpx_track_point, gpx_pt)
            if i == len(data_lst) - 1:
                # Keep the bearing constant for the last track point
                self.records[-1].bearing = self.records[i-1].bearing
        
        self.altitude_df = pd.DataFrame([
            {
                "ts": x.gpx_track_point.time.timestamp(),
                "alt": x.gpx_track_point.elevation
            } for x in self.records
        ])

        self.bearing_df = pd.DataFrame([
            {"ts": x.gpx_track_point.time.timestamp(), 
             "b_x": np.cos(x.bearing.bearing_rad),
             "b_y": np.sin(x.bearing.bearing_rad)} 
                for x in self.records])
    
    def get_altitude_at_time(self, time: datetime) -> int:
        target_ts = time.timestamp()
        return round(np.interp(target_ts, self.altitude_df['ts'], self.altitude_df['alt']))

    def get_bearing_at_time(self, time: datetime) -> Bearing:
        target_ts = time.timestamp()
        if target_ts < self.bearing_df['ts'].min() or target_ts > self.bearing_df['ts'].max():
            return Bearing(0,0)
        x = np.interp(target_ts, self.bearing_df['ts'], self.bearing_df['b_x'])
        y = np.interp(target_ts, self.bearing_df['ts'], self.bearing_df['b_y'])
        bearing_rad = (np.arctan2(y, x) + np.pi) % np.pi
        bearing_deg = np.degrees(bearing_rad)
        return Bearing(bearing_rad, bearing_deg)

@dataclass
class GoProVideoMetadata:

    creation_time: datetime
    number_of_frames: int
    fps: float

    def get_time_at_frame(self, frame: int) -> datetime:
        return self.creation_time + timedelta(seconds=frame/self.fps)
    
    def get_frame_at_time(self, time: datetime) -> int:
        elapsed_seconds = (time - self.creation_time).total_seconds()
        return min(max(round(elapsed_seconds * self.fps), 0), self.number_of_frames - 1)
    
    @classmethod
    def from_video(cls, path_to_video):
        metadata = ffmpeg.probe(path_to_video)['streams'][0]
        metadata['tags']
        nom, denom = metadata['r_frame_rate'].split("/")
        fps = float(nom)/float(denom)

        return GoProVideoMetadata(
            creation_time=datetime.fromisoformat(metadata['tags']['creation_time']),
            number_of_frames=int(metadata['nb_frames']),
            fps = fps
        )


class GoProGPX:

    def __init__(self, path_to_video, keep_poor_trackpoints = False):
        stream = gpmf.io.extract_gpmf_stream(path_to_video)
        gps_blocks = gpmf.gps.extract_gps_blocks(stream)
        gps_data = list(map(gpmf.gps.parse_gps_block, gps_blocks))
        gpx = gpxpy.gpx.GPX()
        gpx_track = gpxpy.gpx.GPXTrack()
        gpx.tracks.append(gpx_track)
        gpx_track.segments.append(gpmf.gps.make_pgx_segment(gps_data))

        assert len(gpx.tracks) == 1, "There are more tracks than expected (>1)"
        assert len(gpx.tracks[0].segments) == 1, "There are more segments than expected (>1)"

        self.track: list[GPXTrackPoint] = gpx.tracks[0].segments[0].points
        if not keep_poor_trackpoints:
            self.track = [x for x in self.track if x.position_dilution < GPX_DILUTION_THRESHOLD]
