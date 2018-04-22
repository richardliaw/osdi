import json
import time


class Timeline(object):
    def __init__(self, tid):
        self.events = []
        self.start_time = time.time()
        self.tid = tid

    def reset(self):
        self.events = []
        self.start_time = time.time()

    def start(self, name):
        self.events.append((self.tid, "B", name, time.time()))

    def end(self, name):
        self.events.append((self.tid, "E", name, time.time()))

    def merge(self, other):
        if other.start_time < self.start_time:
            self.start_time = other.start_time
        self.events.extend(other.events)
        self.events.sort(key=lambda e: e[1])

    def chrome_trace_format(self, filename):
        out = []
        for tid, ph, name, t in self.events:
            ts = int((t - self.start_time) * 1000000)
            out.append({
                "name": name,
                "tid": tid,
                "pid": tid,
                "ph": ph,
                "ts": ts,
            })
        with open(filename, "w") as f:
            f.write(json.dumps(out))
        print("Wrote chrome timeline to", filename)


if __name__ == "__main__":
    a = Timeline(1)
    b = Timeline(2)
    a.start("hi")
    time.sleep(.1)
    b.start("bye")
    a.start("hi3")
    time.sleep(.1)
    a.end("hi3")
    b.end("bye")
    time.sleep(.1)
    a.end("hi")
    b.start("b1")
    b.end("b1")
    a.merge(b)
    a.chrome_trace_format("test.json")
