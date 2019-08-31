def make_image_key(video_id, timestamp):
  """Returns a unique identifier for a video id & timestamp."""
  return "%s,%04d" % (video_id, int(timestamp))

if __name__ == '__main__':
    video_id="aaaa"
    timestamp="930"
    print(make_image_key(video_id,timestamp))