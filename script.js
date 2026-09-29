'use strict';
const cases = {
  'causal-forcing': {name: 'Causal Forcing', duration: 59.8125, label: 'Long video generation', note: 'Follow the subject and scene as the video progresses. Both videos use Causal Forcing, with historical frame selection added on the right.'},
  'longlive2': {name: 'LongLive 2.0', duration: 63.875, label: 'Interactive video generation', note: 'Compare appearance across changing shots and views. Both videos use LongLive 2.0, with historical frame selection added on the right.'},
  'worldmem': {name: 'WorldMem', duration: 2.6, label: 'Action-conditioned world model', note: 'A short world-model comparison. Inspect the scene and ground appearance in the native and FrameMorrow sequences.'}
};
const videos = [document.querySelector('#native-video'), document.querySelector('#ours-video')];
const play = document.querySelector('#play');
const playLabel = document.querySelector('#play-label');
const playIcon = document.querySelector('#play-icon');
const timeline = document.querySelector('#timeline');
const time = document.querySelector('#time');
const speed = document.querySelector('#speed');
const status = document.querySelector('#player-status');
let active = 'causal-forcing';
let revision = 0;
let playing = false;
let loading = false;
let buffering = false;
const clock = seconds => `${Math.floor(seconds / 60)}:${String(Math.floor(seconds % 60)).padStart(2, '0')}`;
function updateTime() {
  const current = videos[0].currentTime || 0;
  timeline.value = current;
  time.textContent = `${clock(current)} / ${clock(Number(timeline.max))}`;
  timeline.setAttribute('aria-valuetext', `${clock(current)} of ${clock(Number(timeline.max))}`);
}
function updateButton() {
  playIcon.textContent = playing ? 'Ⅱ' : '▶';
  playLabel.textContent = loading ? 'Loading…' : playing ? 'Pause comparison' : 'Play comparison';
  play.setAttribute('aria-label', playing ? 'Pause both videos' : 'Play both videos');
  play.disabled = loading;
}
function pause() {
  playing = false;
  buffering = false;
  videos.forEach(video => video.pause());
  updateButton();
}
async function start() {
  const currentRevision = revision;
  if (videos[0].ended || videos[0].currentTime >= Number(timeline.max) - .08) videos.forEach(video => { video.currentTime = 0; });
  loading = true;
  status.textContent = '';
  updateButton();
  try {
    await Promise.all(videos.map(video => video.play()));
    if (currentRevision !== revision) return;
    playing = true;
  } catch (error) {
    if (currentRevision !== revision) return;
    pause();
    status.textContent = 'Playback could not start. Please try Play comparison again.';
  } finally {
    if (currentRevision === revision) { loading = false; updateButton(); }
  }
}
play.addEventListener('click', () => playing ? pause() : start());
timeline.addEventListener('input', () => {
  const value = Number(timeline.value);
  videos.forEach(video => { if (video.readyState > 0) video.currentTime = value; });
  updateTime();
});
speed.addEventListener('change', () => videos.forEach(video => { video.playbackRate = Number(speed.value); }));
for (const video of videos) {
  video.addEventListener('loadedmetadata', () => {
    const durations = videos.map(v => v.duration);
    if (durations.every(Number.isFinite)) { timeline.max = Math.min(...durations); timeline.disabled = false; }
    updateTime();
  });
  video.addEventListener('ended', () => { pause(); updateTime(); });
  video.addEventListener('error', () => { pause(); loading = false; updateButton(); status.textContent = 'This video could not load. Please reload the page or choose another comparison.'; });
  video.addEventListener('waiting', () => {
    if (!playing) return;
    buffering = true;
    videos.forEach(v => v.pause());
    status.textContent = 'Buffering both videos…';
  });
  video.addEventListener('canplay', () => {
    if (playing && buffering && videos.every(v => v.readyState >= 3)) {
      buffering = false;
      status.textContent = '';
      start();
    }
  });
}
videos[0].addEventListener('timeupdate', () => {
  updateTime();
  if (playing && !buffering && Math.abs(videos[0].currentTime - videos[1].currentTime) > .15) videos[1].currentTime = videos[0].currentTime;
});
document.querySelectorAll('[data-case]').forEach(button => button.addEventListener('click', () => {
  if (button.dataset.case === active) return;
  revision++;
  pause();
  loading = false;
  active = button.dataset.case;
  const item = cases[active];
  document.querySelectorAll('[data-case]').forEach(tab => {
    const selected = tab === button;
    tab.classList.toggle('active', selected);
    tab.setAttribute('aria-pressed', String(selected));
  });
  videos.forEach((video, index) => {
    const variant = index === 0 ? 'native' : 'framemorrow';
    video.poster = `assets/${active}-${variant}.jpg`;
    video.querySelector('source').src = `assets/${active}-${variant}.mp4`;
    video.setAttribute('aria-label', `${item.name} ${index === 0 ? 'native generation' : 'with FrameMorrow'}`);
    video.load();
    video.playbackRate = Number(speed.value);
  });
  document.querySelectorAll('.backbone').forEach(label => { label.textContent = item.name; });
  document.querySelector('#case-label').textContent = item.label;
  document.querySelector('#case-note').textContent = item.note;
  timeline.max = item.duration;
  timeline.disabled = true;
  status.textContent = '';
  updateTime();
  updateButton();
}));
document.addEventListener('visibilitychange', () => { if (document.hidden) pause(); });
