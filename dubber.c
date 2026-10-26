/*
 * dubber - offline automatic video dubbing
 *
 *   1. split the soundtrack into voice and background (RoFormer separator)
 *   2. speech recognition on the clean voice (faster-whisper, local model)
 *   3. translate every segment to the target language(s) with a local LLM
 *      that knows the subject of the video and the time each line may
 *      take (NLLB-200 as fallback)
 *   4. synthesize the translated speech (Piper neural TTS, local voice)
 *   5. align each synthesized segment to its original timestamp: speed up
 *      Piper, then time-compress (rubberband, pitch kept) what still does
 *      not fit, so every phrase starts exactly where the original did
 *   6. remux the video: background + new voice (+ translated subtitles)
 *
 * --lipsync replaces 4-5 with dubbing-studio rules: the mouths in the
 * picture are tracked (lips.py), every phrase gets its speaker, sex, face
 * and pitch contour (scene.py), and lipdub.py rewrites lines until their
 * natural spoken length fits the original and, on camera, their visemes
 * follow the lips; each speaker gets a voice of their sex (Kokoro) carrying
 * the original intonation. Nothing is sped up or slowed down.
 *
 * GPU (CUDA) is used for separation, ASR and translation when available;
 * TTS runs on the CPU in parallel, overlapping with the GPU translation of
 * the next language.
 *
 * Everything runs offline once the models are cached (see fetch_models.sh).
 *
 * build: make         usage: ./dubber -i video.mp4 -l en,pt
 */

#define _GNU_SOURCE
#include <errno.h>
#include <getopt.h>
#include <libgen.h>
#include <limits.h>
#include <math.h>
#include <stdarg.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <fcntl.h>
#include <unistd.h>

#define MIN_LENGTH_SCALE 0.80   /* fastest Piper speech (1.25x), still natural */
#define MAX_TEMPO        1.60   /* further time compression by rubberband */
#define SILENCE          300    /* |sample| below this is silence (-41 dBFS) */
#define FADE_MS          8
#define MAX_LANGS        16

typedef struct {
	double start, end;
	char  *src;        /* recognized text */
	char  *dst;        /* translated text (current language) */
	char   wav[PATH_MAX];
	int    nsamples;   /* synthesized length */
} seg_t;

static struct {
	const char *input, *audio, *output, *source, *model;
	const char *voice, *workdir, *device, *prompt, *python, *translator;
	double      bg;
	int         jobs, subs, force, separate, best, lipsync;
} opt = {
	.model = "turbo", .jobs = 0, .subs = 1, .device = "auto", .bg = -1,
	.translator = "auto",
	.separate = 1,
};

static char helper_dir[PATH_MAX];   /* where asr.py etc. live */
static int  retranslate;            /* transcript changed: old translations stale */

/* ------------------------------------------------------------------ */
/* helpers                                                            */
/* ------------------------------------------------------------------ */

static void die(const char *fmt, ...)
{
	va_list ap;
	va_start(ap, fmt);
	fprintf(stderr, "dubber: ");
	vfprintf(stderr, fmt, ap);
	fputc('\n', stderr);
	va_end(ap);
	exit(1);
}

static void step(const char *fmt, ...)
{
	va_list ap;
	va_start(ap, fmt);
	fprintf(stderr, "\n\033[1m==> ");
	vfprintf(stderr, fmt, ap);
	fprintf(stderr, "\033[0m\n");
	va_end(ap);
}

static void *xmalloc(size_t n)
{
	void *p = calloc(1, n ? n : 1);
	if (!p)
		die("out of memory");
	return p;
}

static char *xstrdup(const char *s)
{
	char *p = strdup(s);
	if (!p)
		die("out of memory");
	return p;
}

static int file_ok(const char *path)
{
	struct stat st;
	return stat(path, &st) == 0 && st.st_size > 0;
}

static char *slurp(const char *path, size_t *len)
{
	FILE *f = fopen(path, "rb");
	if (!f)
		return NULL;
	fseek(f, 0, SEEK_END);
	long n = ftell(f);
	fseek(f, 0, SEEK_SET);
	char *buf = xmalloc(n + 1);
	if (fread(buf, 1, n, f) != (size_t)n)
		die("read error on %s", path);
	fclose(f);
	if (len)
		*len = n;
	return buf;
}

static void spit(const char *path, const char *text)
{
	FILE *f = fopen(path, "w");
	if (!f)
		die("cannot write %s", path);
	fputs(text, f);
	fclose(f);
}

/* True if PATH exists and holds exactly TEXT (used to validate reuse). */
static int same_text(const char *path, const char *text)
{
	char *s = slurp(path, NULL);
	int eq = s && !strcmp(s, text);
	free(s);
	return eq;
}

/* Start a child with optional stdin/stdout redirection (no shell). */
static pid_t spawn_ex(char *const argv[], const char *in, const char *out,
		      int quiet)
{
	pid_t pid = fork();
	if (pid < 0)
		die("fork: %s", strerror(errno));
	if (pid == 0) {
		if (quiet) {
			int fd = open("/dev/null", O_WRONLY);
			if (fd >= 0)
				dup2(fd, 2);
		}
		if (in) {
			int fd = open(in, O_RDONLY);
			if (fd < 0 || dup2(fd, 0) < 0)
				_exit(126);
			close(fd);
		}
		if (out) {
			int fd = open(out, O_WRONLY | O_CREAT | O_TRUNC, 0644);
			if (fd < 0 || dup2(fd, 1) < 0)
				_exit(126);
			close(fd);
		}
		execvp(argv[0], argv);
		fprintf(stderr, "dubber: cannot run %s: %s\n", argv[0], strerror(errno));
		_exit(127);
	}
	return pid;
}

static pid_t spawn(char *const argv[], const char *in, const char *out)
{
	return spawn_ex(argv, in, out, 0);
}

/* stdin from a file, stderr discarded (piper is very chatty) */
static pid_t spawn_quiet(char *const argv[], const char *in)
{
	return spawn_ex(argv, in, NULL, 1);
}

/* Children reaped by a wait() that was looking for someone else (the TTS
 * pool reaps the background translator this way); wait_ok() finds them here. */
static struct { pid_t pid; int st; } reaped[16];

static void stash_reaped(pid_t pid, int st)
{
	for (size_t i = 0; i < sizeof reaped / sizeof *reaped; i++)
		if (!reaped[i].pid) {
			reaped[i].pid = pid;
			reaped[i].st = st;
			return;
		}
	die("too many background processes");
}

static int wait_ok(pid_t pid)
{
	int st;
	for (size_t i = 0; i < sizeof reaped / sizeof *reaped; i++)
		if (reaped[i].pid == pid) {
			reaped[i].pid = 0;
			st = reaped[i].st;
			return WIFEXITED(st) && WEXITSTATUS(st) == 0;
		}
	while (waitpid(pid, &st, 0) < 0)
		if (errno != EINTR)
			return 0;
	return WIFEXITED(st) && WEXITSTATUS(st) == 0;
}

static void run(char *const argv[], const char *in, const char *out)
{
	if (!wait_ok(spawn(argv, in, out)))
		die("command failed: %s", argv[0]);
}

/* Run a command and return its stdout, or NULL if it fails. */
static char *capture_try(char *const argv[])
{
	char tmp[PATH_MAX];
	snprintf(tmp, sizeof tmp, "%s/.capture", opt.workdir);
	int ok = wait_ok(spawn_ex(argv, NULL, tmp, 1));
	char *s = slurp(tmp, NULL);
	unlink(tmp);
	if (!ok) {
		free(s);
		return NULL;
	}
	return s ? s : xstrdup("");
}

static char *capture(char *const argv[])
{
	char *s = capture_try(argv);
	if (!s)
		die("command failed: %s", argv[0]);
	return s;
}

static int in_path(const char *prog)
{
	if (strchr(prog, '/'))
		return access(prog, X_OK) == 0;
	const char *path = getenv("PATH");
	if (!path)
		return 0;
	char *dup = xstrdup(path), *save, *dir;
	char buf[PATH_MAX];
	int found = 0;
	for (dir = strtok_r(dup, ":", &save); dir && !found;
	     dir = strtok_r(NULL, ":", &save)) {
		snprintf(buf, sizeof buf, "%s/%s", dir, prog);
		found = access(buf, X_OK) == 0;
	}
	free(dup);
	return found;
}

static char *trim(char *s)
{
	while (*s == ' ' || *s == '\t' || *s == '\r' || *s == '\n')
		s++;
	char *e = s + strlen(s);
	while (e > s && (e[-1] == ' ' || e[-1] == '\t' || e[-1] == '\r' || e[-1] == '\n'))
		*--e = 0;
	return s;
}

static const char *home(void)
{
	const char *h = getenv("HOME");
	return h ? h : ".";
}

static const char *helper(const char *name)
{
	static char buf[4][PATH_MAX];
	static int k;
	char *p = buf[k++ & 3];
	snprintf(p, PATH_MAX, "%s/%s", helper_dir, name);
	if (!file_ok(p))
		die("helper %s not found next to the dubber binary", p);
	return p;
}

/* ------------------------------------------------------------------ */
/* environment: python interpreter, GPU                               */
/* ------------------------------------------------------------------ */

static void pick_python(void)
{
	static char venv[PATH_MAX];
	if ((opt.python = getenv("DUBBER_PYTHON")))
		return;
	snprintf(venv, sizeof venv, "%s/.local/share/dubber/venv/bin/python", home());
	opt.python = access(venv, X_OK) == 0 ? venv : "python3";
}

/* Resolve --device auto into cuda/cpu and export it for the helpers. */
static void pick_device(void)
{
	step("compute device");
	if (strcmp(opt.device, "auto") && strcmp(opt.device, "cuda") &&
	    strcmp(opt.device, "cpu"))
		die("--device must be auto, cuda or cpu");

	char *gpu = NULL;
	if (strcmp(opt.device, "cpu") && in_path("nvidia-smi")) {
		char *argv[] = { "nvidia-smi", "--query-gpu=name,memory.total",
				 "--format=csv,noheader", NULL };
		gpu = capture_try(argv);
		if (gpu && !*trim(gpu)) {
			free(gpu);
			gpu = NULL;
		}
	}
	/* the GPU must also be usable from the python stack */
	int torch_cuda = 0;
	if (gpu) {
		char *argv[] = { (char *)opt.python, "-c",
				 "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)",
				 NULL };
		char *s = capture_try(argv);
		torch_cuda = s != NULL;
		free(s);
	}

	if (gpu && torch_cuda) {
		fprintf(stderr, "    GPU: %s -> separation, ASR and translation on CUDA,\n"
			"    TTS on %d CPU workers\n", trim(gpu), opt.jobs);
		opt.device = "cuda";
	} else {
		if (!strcmp(opt.device, "cuda"))
			die("--device cuda requested but %s",
			    gpu ? "PyTorch in this python has no CUDA support"
				: "no NVIDIA GPU found");
		if (gpu)
			fprintf(stderr, "    GPU found (%s) but %s has no CUDA PyTorch;"
				" run fetch_models.sh to set up the venv\n",
				trim(gpu), opt.python);
		fprintf(stderr, "    CPU only (%d TTS workers)\n", opt.jobs);
		opt.device = "cpu";
	}
	free(gpu);
	setenv("DUBBER_DEVICE", opt.device, 1);
}

/* ------------------------------------------------------------------ */
/* voices (no network: everything must already be on disk)            */
/* ------------------------------------------------------------------ */

static const struct { const char *lang, *voice; } voices[] = {
	{ "en", "en_US-lessac-medium" },   { "pt", "pt_BR-faber-medium" },
	{ "fr", "fr_FR-siwis-medium" },    { "es", "es_ES-davefx-medium" },
	{ "de", "de_DE-thorsten-medium" }, { "it", "it_IT-paola-medium" },
	{ "zh", "zh_CN-huayan-medium" },   { "ru", "ru_RU-irina-medium" },
	{ "hi", "hi_IN-pratham-medium" },  { "ar", "ar_JO-kareem-medium" },
	{ "nl", "nl_NL-mls-medium" },      { "pl", "pl_PL-gosia-medium" },
	{ "tr", "tr_TR-dfki-medium" },     { "uk", "uk_UA-ukrainian_tts-medium" },
	{ "vi", "vi_VN-vais1000-medium" },
};

static void find_voice(const char *lang, char *path, size_t n)
{
	if (opt.voice) {
		snprintf(path, n, "%s", opt.voice);
	} else {
		const char *dir = getenv("DUBBER_VOICES");
		char def[PATH_MAX];
		if (!dir) {
			snprintf(def, sizeof def, "%s/.local/share/dubber/voices", home());
			dir = def;
		}
		const char *name = NULL;
		for (size_t i = 0; i < sizeof voices / sizeof *voices; i++)
			if (!strcmp(voices[i].lang, lang))
				name = voices[i].voice;
		if (!name)
			die("no default voice for '%s'; pass --voice model.onnx", lang);
		snprintf(path, n, "%s/%s.onnx", dir, name);
	}
	if (!file_ok(path))
		die("voice model %s not found; run ./fetch_models.sh %s %s",
		    path, opt.model, lang);
}

/* ------------------------------------------------------------------ */
/* SRT                                                                */
/* ------------------------------------------------------------------ */

static double srt_time(const char *s)
{
	int h, m, sec, ms;
	if (sscanf(s, "%d:%d:%d%*[,.]%d", &h, &m, &sec, &ms) != 4)
		return -1;
	return h * 3600.0 + m * 60.0 + sec + ms / 1000.0;
}

static seg_t *parse_srt(const char *path, int *count)
{
	char *buf = slurp(path, NULL);
	if (!buf)
		die("cannot read %s", path);

	int cap = 256, n = 0;
	seg_t *segs = xmalloc(cap * sizeof *segs);
	char *rest = buf, *line;

	/* cue = index line, "start --> end" line, text lines, blank line */
	while ((line = strsep(&rest, "\n"))) {
		char *arrow = strstr(line, "-->");
		if (!arrow)
			continue;
		double a = srt_time(trim(line)), b = srt_time(trim(arrow + 3));
		char text[8192] = "";
		while ((line = strsep(&rest, "\n")) && *trim(line)) {
			if (*text)
				strncat(text, " ", sizeof text - strlen(text) - 1);
			strncat(text, trim(line), sizeof text - strlen(text) - 1);
		}
		if (a >= 0 && b >= a && *trim(text)) {
			if (n == cap)
				segs = realloc(segs, (cap *= 2) * sizeof *segs);
			memset(&segs[n], 0, sizeof *segs);
			segs[n].start = a;
			segs[n].end = b;
			segs[n].src = xstrdup(trim(text));
			n++;
		}
	}
	free(buf);
	*count = n;
	return segs;
}

static void fmt_srt_time(char *out, double t)
{
	long ms = lround(t * 1000);
	sprintf(out, "%02ld:%02ld:%02ld,%03ld",
		ms / 3600000, ms / 60000 % 60, ms / 1000 % 60, ms % 1000);
}

static void write_srt(const char *path, seg_t *s, int n)
{
	FILE *f = fopen(path, "w");
	if (!f)
		die("cannot write %s", path);
	for (int i = 0, k = 1; i < n; i++) {
		if (!*s[i].dst)
			continue;
		char a[32], b[32];
		fmt_srt_time(a, s[i].start);
		fmt_srt_time(b, s[i].end);
		fprintf(f, "%d\n%s --> %s\n%s\n\n", k++, a, b, s[i].dst);
	}
	fclose(f);
}

/* ------------------------------------------------------------------ */
/* WAV (16-bit PCM)                                                   */
/* ------------------------------------------------------------------ */

static int16_t *read_wav(const char *path, int *nsamples, int *rate)
{
	size_t len;
	uint8_t *b = (uint8_t *)slurp(path, &len);
	if (!b || len < 12 || memcmp(b, "RIFF", 4) || memcmp(b + 8, "WAVE", 4))
		die("%s: not a WAV file", path);

	int channels = 0, bits = 0;
	size_t p = 12;
	*rate = 0;
	while (p + 8 <= len) {
		uint32_t sz = b[p + 4] | b[p + 5] << 8 | b[p + 6] << 16 | (uint32_t)b[p + 7] << 24;
		uint8_t *c = b + p + 8;
		if (!memcmp(b + p, "fmt ", 4)) {
			channels = c[2] | c[3] << 8;
			*rate = c[4] | c[5] << 8 | c[6] << 16 | c[7] << 24;
			bits = c[14] | c[15] << 8;
		} else if (!memcmp(b + p, "data", 4)) {
			if (sz > len - p - 8)       /* streamed WAVs may lie */
				sz = len - p - 8;
			if (channels < 1 || bits != 16)
				die("%s: need 16-bit PCM", path);
			int frames = sz / (2 * channels);
			int16_t *out = xmalloc(frames * sizeof *out);
			for (int i = 0; i < frames; i++) {   /* downmix to mono */
				int acc = 0;
				for (int ch = 0; ch < channels; ch++) {
					uint8_t *s = c + 2 * (i * channels + ch);
					acc += (int16_t)(s[0] | s[1] << 8);
				}
				out[i] = acc / channels;
			}
			free(b);
			*nsamples = frames;
			return out;
		}
		p += 8 + sz + (sz & 1);
	}
	die("%s: no data chunk", path);
	return NULL;
}

static void write_wav(const char *path, const int16_t *pcm, long n, int rate)
{
	FILE *f = fopen(path, "wb");
	if (!f)
		die("cannot write %s", path);
	uint32_t data = n * 2, riff = 36 + data, fmt = 16, br = rate * 2;
	uint16_t pcmfmt = 1, ch = 1, align = 2, bits = 16;
	fwrite("RIFF", 1, 4, f); fwrite(&riff, 4, 1, f); fwrite("WAVE", 1, 4, f);
	fwrite("fmt ", 1, 4, f); fwrite(&fmt, 4, 1, f);
	fwrite(&pcmfmt, 2, 1, f); fwrite(&ch, 2, 1, f);
	fwrite(&rate, 4, 1, f);   fwrite(&br, 4, 1, f);
	fwrite(&align, 2, 1, f);  fwrite(&bits, 2, 1, f);
	fwrite("data", 1, 4, f);  fwrite(&data, 4, 1, f);
	fwrite(pcm, 2, n, f);
	fclose(f);
}

/* ------------------------------------------------------------------ */
/* pipeline steps                                                     */
/* ------------------------------------------------------------------ */

static double media_duration(const char *path)
{
	char *argv[] = { "ffprobe", "-v", "error", "-show_entries", "format=duration",
			 "-of", "default=nw=1:nk=1", (char *)path, NULL };
	char *s = capture(argv);
	double d = atof(s);
	free(s);
	return d;
}

static int has_audio(const char *path)
{
	char *argv[] = { "ffprobe", "-v", "error", "-select_streams", "a",
			 "-show_entries", "stream=index", "-of", "csv=p=0",
			 (char *)path, NULL };
	char *s = capture(argv);
	int yes = *trim(s) != 0;
	free(s);
	return yes;
}

/* Decode any container/codec to 44.1 kHz stereo PCM WAV (the separator's
 * native format; its loader cannot read m4a/aac). */
static void extract_audio(const char *src, const char *wav)
{
	char key[PATH_MAX];
	snprintf(key, sizeof key, "%s/source.key", opt.workdir);
	step("extracting audio from %s", src);
	if (!opt.force && file_ok(wav) && same_text(key, src)) {
		fprintf(stderr, "    reusing %s\n", wav);
		return;
	}
	char *argv[] = { "ffmpeg", "-y", "-v", "error", "-stats", "-i", (char *)src,
			 "-vn", "-ac", "2", "-ar", "44100", "-c:a", "pcm_s16le",
			 (char *)wav, NULL };
	run(argv, NULL, NULL);
	spit(key, src);
}

/* Split SPEECH into workdir/vocals.wav + workdir/background.wav. */
static void separate(const char *speech)
{
	char voc[PATH_MAX], bg[PATH_MAX], key[PATH_MAX];
	snprintf(voc, sizeof voc, "%s/vocals.wav", opt.workdir);
	snprintf(bg, sizeof bg, "%s/background.wav", opt.workdir);
	snprintf(key, sizeof key, "%s/separate.key", opt.workdir);

	step("separating voice from background (RoFormer, %s)", opt.device);
	if (!opt.force && file_ok(voc) && file_ok(bg) && same_text(key, speech)) {
		fprintf(stderr, "    reusing %s\n", bg);
		return;
	}
	char *argv[] = { (char *)opt.python, (char *)helper("separate.py"),
			 (char *)speech, (char *)opt.workdir, NULL };
	run(argv, NULL, NULL);
	spit(key, speech);
}

/* Returns detected/forced source language (static buffer). */
static const char *transcribe(const char *audio, char *srt, size_t n)
{
	static char lang[16];
	char json[PATH_MAX], keyf[PATH_MAX], key[8192];
	snprintf(srt, n, "%s/orig.srt", opt.workdir);
	snprintf(json, sizeof json, "%s/orig.json", opt.workdir);
	snprintf(keyf, sizeof keyf, "%s/asr.key", opt.workdir);
	snprintf(key, sizeof key, "words\n%s\n%s\n%s\n%s\n", opt.model,
		 opt.source ? opt.source : "auto", audio,
		 opt.prompt ? opt.prompt : "");

	step("speech recognition (faster-whisper %s, %s)", opt.model, opt.device);
	if (opt.force || !file_ok(srt) || !file_ok(json) || !same_text(keyf, key)) {
		char *argv[] = { (char *)opt.python, (char *)helper("asr.py"),
				 (char *)opt.model,
				 (char *)(opt.source ? opt.source : "auto"),
				 (char *)audio, (char *)opt.workdir,
				 (char *)(opt.prompt ? opt.prompt : ""), NULL };
		run(argv, NULL, NULL);
		spit(keyf, key);
	} else {
		fprintf(stderr, "    reusing %s\n", srt);
	}

	if (opt.source) {
		snprintf(lang, sizeof lang, "%s", opt.source);
	} else {   /* asr.py writes {"language": "xx", ...} */
		char *j = slurp(json, NULL), *p;
		if (!j || !(p = strstr(j, "\"language\"")) ||
		    sscanf(p, "\"language\" : \"%15[^\"]\"", lang) != 1)
			die("cannot detect source language; pass -s");
		free(j);
		fprintf(stderr, "    detected language: %s\n", lang);
	}
	return lang;
}

/* --lipsync: track every face's mouth in the background (CPU), while the
 * GPU separates and transcribes. Returns 0 when lips.npz is up to date. */
static pid_t lips_start(void)
{
	char npz[PATH_MAX], key[PATH_MAX];
	snprintf(npz, sizeof npz, "%s/lips.npz", opt.workdir);
	snprintf(key, sizeof key, "%s/lips.key", opt.workdir);
	if (!opt.force && file_ok(npz) && same_text(key, opt.input))
		return 0;
	unlink(key);
	char log[PATH_MAX];
	snprintf(log, sizeof log, "%s/lips.log", opt.workdir);
	char *argv[] = { (char *)opt.python, (char *)helper("lips.py"),
			 (char *)opt.input, npz, NULL };
	return spawn(argv, NULL, log);
}

static void lips_finish(pid_t pid)
{
	char npz[PATH_MAX], key[PATH_MAX];
	snprintf(npz, sizeof npz, "%s/lips.npz", opt.workdir);
	snprintf(key, sizeof key, "%s/lips.key", opt.workdir);
	step("lip tracking (MediaPipe Face Landmarker, CPU)");
	if (!pid) {
		fprintf(stderr, "    reusing %s\n", npz);
		return;
	}
	if (!wait_ok(pid))
		die("lip tracking failed (see %s/lips.log)", opt.workdir);
	spit(key, opt.input);
	char log[PATH_MAX], *t;
	snprintf(log, sizeof log, "%s/lips.log", opt.workdir);
	if ((t = slurp(log, NULL))) {     /* its last line is the summary */
		char *e = t + strlen(t);
		while (e > t && e[-1] == '\n')
			*--e = 0;
		char *l = strrchr(t, '\n');
		fprintf(stderr, "    %s\n", l ? l + 1 : t);
		free(t);
	}
}

/* Speaker turns, sex, speaking face and intonation of every phrase;
 * returns scene.srt, the phrases split at speaker turns. */
static void scene(char *srt, size_t n)
{
	step("speakers, faces and intonation");
	char *argv[] = { (char *)opt.python, (char *)helper("scene.py"),
			 (char *)opt.workdir, NULL };
	run(argv, NULL, NULL);
	snprintf(srt, n, "%s/scene.srt", opt.workdir);
}

/* Languages lipdub.py has Kokoro voices for. */
static int kokoro_lang(const char *lang)
{
	static const char *k[] = { "pt", "en", "es", "fr", "it", "hi" };
	for (size_t i = 0; i < sizeof k / sizeof *k; i++)
		if (!strcmp(k[i], lang))
			return 1;
	return 0;
}

/* Adapt the translation to time and lips, voice it with intonation;
 * the chosen wording replaces s[].dst (for the subtitles). */
static void lipdub(seg_t *s, int n, const char *src, const char *lang)
{
	step("lip-sync adaptation and voices %s (Kokoro, %s)", lang,
	     opt.device);
	char *argv[] = { (char *)opt.python, (char *)helper("lipdub.py"),
			 (char *)opt.workdir, (char *)src, (char *)lang, NULL };
	run(argv, NULL, NULL);
	char path[PATH_MAX];
	snprintf(path, sizeof path, "%s/%s_dub.txt", opt.workdir, lang);
	char *buf = slurp(path, NULL), *line = buf;
	if (!buf)
		die("cannot read %s", path);
	for (int i = 0; i < n; i++) {
		char *nl = line ? strchr(line, '\n') : NULL;
		if (!line || !nl)
			die("%s has fewer lines than segments", path);
		*nl = 0;
		free(s[i].dst);
		s[i].dst = xstrdup(trim(line));
		line = nl + 1;
	}
	free(buf);
}

/* Start translating src.txt to LANG in the background (GPU), so the CPU
 * can keep synthesizing the previous language meanwhile. Returns 0 when
 * nothing has to run. */
static pid_t translate_start(const char *src, const char *lang)
{
	char in[PATH_MAX], out[PATH_MAX], slots[PATH_MAX];
	snprintf(in, sizeof in, "%s/src.txt", opt.workdir);
	snprintf(out, sizeof out, "%s/%s.txt", opt.workdir, lang);
	snprintf(slots, sizeof slots, "%s/slots.txt", opt.workdir);
	if (!strcmp(src, lang) || (!opt.force && !retranslate && file_ok(out)))
		return 0;
	unlink(out);
	char *argv[] = { (char *)opt.python, (char *)helper("translate.py"),
			 (char *)src, (char *)lang, in, out, slots, NULL };
	return spawn(argv, NULL, NULL);
}

static void translate_finish(pid_t pid, seg_t *s, int n, const char *src,
			     const char *lang)
{
	char out[PATH_MAX];
	snprintf(out, sizeof out, "%s/%s.txt", opt.workdir, lang);

	step("translation %s -> %s (%s, %s)", src, lang, opt.translator, opt.device);
	for (int i = 0; i < n; i++)
		free(s[i].dst);
	if (!strcmp(src, lang)) {
		for (int i = 0; i < n; i++)
			s[i].dst = xstrdup(s[i].src);
		return;
	}
	if (pid > 0 && !wait_ok(pid))
		die("translation to %s failed", lang);
	if (!pid)
		fprintf(stderr, "    reusing %s\n", out);

	char *buf = slurp(out, NULL), *line = buf;
	if (!buf)
		die("cannot read %s", out);
	for (int i = 0; i < n; i++) {
		char *nl = line ? strchr(line, '\n') : NULL;
		if (!line || !nl)
			die("translator returned fewer lines than segments");
		*nl = 0;
		s[i].dst = xstrdup(trim(line));
		line = nl + 1;
	}
	free(buf);
}

/* Synthesize the segments listed in idx[] at the given length scales. */
static void synth_batch(seg_t *s, int *idx, double *scale, int count,
			const char *voice, const char *dir)
{
	pid_t *pids = xmalloc(opt.jobs * sizeof *pids);
	int   *who = xmalloc(opt.jobs * sizeof *who);
	int running = 0, next = 0, done = 0;

	while (done < count) {
		while (running < opt.jobs && next < count) {
			int i = idx[next];
			char txt[PATH_MAX], ls[32];
			snprintf(txt, sizeof txt, "%s/%05d.txt", dir, i);
			snprintf(s[i].wav, sizeof s[i].wav, "%s/%05d.wav", dir, i);
			snprintf(ls, sizeof ls, "%.3f", scale[next]);
			spit(txt, s[i].dst);
			char *argv[] = { "piper", "-m", (char *)voice, "-f", s[i].wav,
					 "--length-scale", ls, "--sentence-silence", "0",
					 NULL };
			pids[running] = spawn_quiet(argv, txt);
			who[running] = i;
			running++;
			next++;
		}
		int st, mine = 0;
		pid_t pid = wait(&st);
		if (pid < 0) {
			if (errno == EINTR)
				continue;
			die("wait: %s", strerror(errno));
		}
		for (int k = 0; k < running; k++)
			mine |= pids[k] == pid;
		if (!mine) {            /* e.g. the background translator */
			stash_reaped(pid, st);
			continue;
		}
		for (int k = 0; k < running; k++) {
			if (pids[k] != pid)
				continue;
			if (!WIFEXITED(st) || WEXITSTATUS(st) || !file_ok(s[who[k]].wav)) {
				fprintf(stderr, "\n    warning: piper failed on segment %d, "
					"skipping: \"%.80s\"\n", who[k], s[who[k]].dst);
				s[who[k]].dst[0] = 0;   /* leaves a silent gap */
			}
			pids[k] = pids[running - 1];
			who[k] = who[running - 1];
			running--;
			done++;
			fprintf(stderr, "\r    %d/%d", done, count);
			break;
		}
	}
	fputc('\n', stderr);
	free(pids);
	free(who);
}

/* Text worth sending to TTS: has a letter/digit (any non-ASCII counts). */
static int speakable(char *t)
{
	for (unsigned char *p = (unsigned char *)t; *p; p++)
		if (*p >= 0x80 || (*p >= '0' && *p <= '9') ||
		    ((*p | 0x20) >= 'a' && (*p | 0x20) <= 'z'))
			return 1;
	*t = 0;   /* punctuation only: treat as silence */
	return 0;
}

/* Speech part of a clip: skip the silence Piper puts around it, so the
 * first word lands exactly on the original start time. */
static void speech_bounds(const int16_t *pcm, int ns, int rate, int *a, int *b)
{
	int pad = rate / 200;   /* keep 5 ms around the speech */
	int i = 0, j = ns;
	while (i < ns && abs(pcm[i]) < SILENCE)
		i++;
	while (j > i && abs(pcm[j - 1]) < SILENCE)
		j--;
	*a = i > pad ? i - pad : 0;
	*b = j + pad < ns ? j + pad : ns;
	if (*b <= *a) {
		*a = 0;
		*b = ns;
	}
}

/* Seconds of actual speech in a synthesized clip. */
static double speech_len(const char *wav, int *rate)
{
	int ns, a, b;
	int16_t *pcm = read_wav(wav, &ns, rate);
	speech_bounds(pcm, ns, *rate, &a, &b);
	free(pcm);
	return (double)(b - a) / *rate;
}

/* Time-compress a clip by TEMPO keeping the pitch (rubberband). */
static void compress(seg_t *seg, double tempo, int rate)
{
	char trim[PATH_MAX + 8], fit[PATH_MAX + 8], af[64], ar[16];
	int ns, a, b;
	int16_t *pcm = read_wav(seg->wav, &ns, &rate);
	speech_bounds(pcm, ns, rate, &a, &b);
	snprintf(trim, sizeof trim, "%.*s_trim.wav", (int)strlen(seg->wav) - 4, seg->wav);
	snprintf(fit, sizeof fit, "%.*s_fit.wav", (int)strlen(seg->wav) - 4, seg->wav);
	write_wav(trim, pcm + a, b - a, rate);
	free(pcm);
	snprintf(af, sizeof af, "rubberband=tempo=%.4f:window=short", tempo);
	snprintf(ar, sizeof ar, "%d", rate);
	char *argv[] = { "ffmpeg", "-y", "-v", "error", "-i", trim, "-af", af,
			 "-ac", "1", "-ar", ar, "-c:a", "pcm_s16le", fit, NULL };
	run(argv, NULL, NULL);
	unlink(trim);
	snprintf(seg->wav, sizeof seg->wav, "%s", fit);
}

/* Time available to segment i: until the next one starts. */
static double slot_of(seg_t *s, int n, int i, double total)
{
	return (i + 1 < n ? s[i + 1].start : total) - s[i].start;
}

/* Add CNT samples of PCM to TRACK at AT, with FADE-sample ramps at both
 * ends and clipping protection. */
static void mix_clip(int16_t *track, long len, const int16_t *pcm, long cnt,
		     long at, int fade)
{
	if (at < 0) {
		pcm -= at;
		cnt += at;
		at = 0;
	}
	if (at + cnt > len)
		cnt = len - at;
	for (long t = 0; t < cnt; t++) {
		double g = 1.0;
		if (t < fade)
			g = (double)t / fade;
		else if (t >= cnt - fade)
			g = (double)(cnt - 1 - t) / fade;
		int v = track[at + t] + (int)(pcm[t] * g);
		track[at + t] = v > 32767 ? 32767 : v < -32768 ? -32768 : v;
	}
}

/* --lipsync: lay out lipdub.py's clips at the times it chose
 * (place_LANG.txt: "start clip" per line). Clips are used as spoken:
 * no speed change; only a clip that would run into the next is faded out. */
static void place(seg_t *s, int n, double total, const char *lang,
		  const char *outwav)
{
	char path[PATH_MAX];
	snprintf(path, sizeof path, "%s/place_%s.txt", opt.workdir, lang);
	step("building dubbed voice track");
	char *buf = slurp(path, NULL);
	if (!buf)
		die("cannot read %s", path);
	int cap = n + 1, m = 0;
	double *at = xmalloc(cap * sizeof *at);
	char **wav = xmalloc(cap * sizeof *wav);
	for (char *save, *l = strtok_r(buf, "\n", &save); l;
	     l = strtok_r(NULL, "\n", &save)) {
		char *sp = strchr(l, ' ');
		if (!sp || m == cap)
			continue;
		*sp = 0;
		at[m] = atof(l);
		wav[m++] = sp + 1;
	}
	int rate = 0, cut = 0, early = 0;
	long len = 0;
	int16_t *track = NULL;
	for (int k = 0; k < m; k++) {
		int ns, r;
		int16_t *pcm = read_wav(wav[k], &ns, &r);
		if (!track) {
			rate = r;
			len = (long)(total * rate) + 1;
			track = xmalloc(len * sizeof *track);   /* zeroed */
		}
		if (r != rate)
			die("%s: sample rate mismatch", wav[k]);
		long room = k + 1 < m ? (long)((at[k + 1] - at[k]) * rate) : ns;
		long cnt = ns;
		int fade = rate * FADE_MS / 1000;
		if (cnt > room + fade) {
			cnt = room;
			cut++;
		}
		mix_clip(track, len, pcm, cnt, (long)(at[k] * rate), fade);
		free(pcm);
	}
	for (int i = 0, k = 0; i < n && k < m; i++)   /* phrases started early */
		if (*s[i].dst && speakable(s[i].dst))
			early += at[k++] < s[i].start - 0.02;
	if (!track)
		die("no segment could be synthesized");
	fprintf(stderr, "    %d clips at their natural speed; %d off-screen lines "
		"start early", m, early);
	if (cut)
		fprintf(stderr, ", %d faded out before the next line", cut);
	fputc('\n', stderr);
	write_wav(outwav, track, len, rate);
	free(track);
	free(at);
	free(wav);
	free(buf);
}

static void synthesize(seg_t *s, int n, double total, const char *voice,
		       const char *lang, const char *outwav)
{
	char dir[PATH_MAX];
	snprintf(dir, sizeof dir, "%s/tts_%s", opt.workdir, lang);
	mkdir(dir, 0755);

	int *idx = xmalloc(n * sizeof *idx), m = 0;
	double *scale = xmalloc(n * sizeof *scale);
	for (int i = 0; i < n; i++)
		if (speakable(s[i].dst)) {
			idx[m] = i;
			scale[m++] = 1.0;
		}

	step("text-to-speech %s: %d segments (piper, %d CPU workers)", lang, m, opt.jobs);
	synth_batch(s, idx, scale, m, voice, dir);

	/* Pass 2: let Piper speak faster where the speech overruns its slot. */
	int rate = 0, r, k = 0;
	for (int j = 0; j < m; j++) {
		int i = idx[j];
		if (!*s[i].dst)
			continue;
		double dur = speech_len(s[i].wav, &r), slot = slot_of(s, n, i, total);
		if (!rate)
			rate = r;
		if (dur > slot) {
			double ls = slot / dur;
			idx[k] = i;
			scale[k++] = ls < MIN_LENGTH_SCALE ? MIN_LENGTH_SCALE : ls;
		}
	}
	if (k) {
		step("re-synthesizing %d segments faster to fit their time slot", k);
		synth_batch(s, idx, scale, k, voice, dir);
	}
	if (!rate)
		die("no segment could be synthesized");

	/* Pass 3: time-compress (pitch kept) what Piper alone cannot fit. */
	int squeezed = 0, cut = 0;
	for (int j = 0; j < k; j++) {
		int i = idx[j];
		if (!*s[i].dst)
			continue;
		double dur = speech_len(s[i].wav, &r), slot = slot_of(s, n, i, total);
		if (dur <= slot)
			continue;
		double tempo = dur / slot;
		compress(&s[i], tempo < MAX_TEMPO ? tempo : MAX_TEMPO, rate);
		squeezed++;
		fprintf(stderr, "\r    time-compressed %d", squeezed);
	}
	if (squeezed)
		fputc('\n', stderr);

	/* Lay the segments out: each starts exactly at its original time. */
	step("building dubbed voice track");
	long len = (long)(total * rate) + 1;
	int16_t *track = xmalloc(len * sizeof *track);   /* zeroed */
	int fade = rate * FADE_MS / 1000;
	for (int i = 0; i < n; i++) {
		if (!*s[i].dst)
			continue;
		int ns, a, b;
		int16_t *pcm = read_wav(s[i].wav, &ns, &r);
		if (r != rate)
			die("%s: sample rate mismatch", s[i].wav);
		speech_bounds(pcm, ns, rate, &a, &b);
		long at = (long)(s[i].start * rate);
		long room = (long)(slot_of(s, n, i, total) * rate);
		long cnt = b - a;
		if (cnt > room + fade) {   /* beyond MAX_TEMPO: fade out early */
			cnt = room;
			cut++;
		}
		mix_clip(track, len, pcm + a, cnt, at, fade);
		free(pcm);
	}
	fprintf(stderr, "    all %d segments start on time: %d sped up by Piper, "
		"%d also time-compressed", m, k, squeezed);
	if (cut)
		fprintf(stderr, ", %d cut short (would need > %.1fx)",
			cut, 1 / MIN_LENGTH_SCALE * MAX_TEMPO);
	fputc('\n', stderr);
	write_wav(outwav, track, len, rate);
	free(track);
	free(idx);
	free(scale);
}

/* Video from the input, dubbed voice mixed over BGSRC (if any). */
static void mux(const char *output, const char *dubwav, const char *srt,
		const char *bgsrc, double bgvol)
{
	step("writing %s", output);
	char filter[256], *argv[48];
	int a = 0, inputs = 2;

	argv[a++] = "ffmpeg"; argv[a++] = "-y"; argv[a++] = "-v"; argv[a++] = "error";
	argv[a++] = "-stats";
	argv[a++] = "-i"; argv[a++] = (char *)opt.input;
	argv[a++] = "-i"; argv[a++] = (char *)dubwav;
	if (bgsrc) {
		argv[a++] = "-i"; argv[a++] = (char *)bgsrc;
		inputs++;
	}
	if (opt.subs) {
		argv[a++] = "-i"; argv[a++] = (char *)srt;
	}
	argv[a++] = "-map"; argv[a++] = "0:v:0";
	if (bgsrc) {
		/* stereo background + centered voice, limiter against clipping */
		snprintf(filter, sizeof filter,
			 "[2:a:0]volume=%.3f[bg];[1:a][bg]amix=inputs=2:duration=first:"
			 "normalize=0,alimiter=limit=0.95[a]", bgvol);
		argv[a++] = "-filter_complex"; argv[a++] = filter;
		argv[a++] = "-map"; argv[a++] = "[a]";
	} else {
		argv[a++] = "-map"; argv[a++] = "1:a";
	}
	if (opt.subs) {
		static char map[16];
		snprintf(map, sizeof map, "%d:s", inputs);
		argv[a++] = "-map"; argv[a++] = map;
		argv[a++] = "-c:s"; argv[a++] = "mov_text";
	}
	argv[a++] = "-c:v"; argv[a++] = "copy";
	argv[a++] = "-c:a"; argv[a++] = "aac";
	argv[a++] = "-b:a"; argv[a++] = "192k";
	argv[a++] = "-movflags"; argv[a++] = "+faststart";
	argv[a++] = (char *)output;
	argv[a] = NULL;
	run(argv, NULL, NULL);
}

/* ------------------------------------------------------------------ */

static void usage(int code)
{
	fprintf(code ? stderr : stdout,
"usage: dubber -i VIDEO -l LANG[,LANG...] [options]\n"
"\n"
"Offline dubbing: RoFormer (voice/background split) -> faster-whisper (ASR)\n"
"-> local LLM or NLLB-200 (translation) -> Piper (TTS), background preserved.\n"
"\n"
"  -i, --input FILE     input video\n"
"  -l, --lang LIST      target language(s), comma separated:\n"
"                       en pt fr es de it zh ru hi ar nl pl tr uk vi\n"
"  -o, --output FILE    output video (default: <input>_<lang>.mp4); with\n"
"                       several languages _<lang> is added to the name\n"
"  -a, --audio FILE     take sound from this file instead of the video\n"
"                       (for video-only files such as YouTube DASH streams)\n"
"  -s, --source CODE    source language, e.g. ur hi en (default: auto-detect)\n"
"  -m, --model NAME     whisper model: small medium large-v3 turbo (default)\n"
"  -p, --prompt TEXT    topic and technical words, given to speech recognition\n"
"                       and to the translator, e.g. \"CNC fiber laser build\"\n"
"  -t, --translator T   auto (default: llm if installed), llm or nllb\n"
"      --best           understand the source with Gemma 3 12B (better terms,\n"
"                       ~10x slower on a 4 GB GPU; DUBBER_LLM_SIZE=12b\n"
"                       ./fetch_models.sh downloads it)\n"
"  -v, --voice FILE     Piper voice .onnx (default: per-language voice)\n"
"  -d, --device DEV     auto (default), cuda or cpu\n"
"  -b, --bg VOLUME      background volume 0..2 (default 1.0 = as recorded;\n"
"                       with --no-separate: original audio, default 0)\n"
"      --no-separate    skip voice/background separation\n"
"      --lipsync        dub like a studio: track the mouths in the picture,\n"
"                       reword lines until they fit the time and the lips\n"
"                       (never time-stretched), one voice per speaker of\n"
"                       their sex, original intonation (pt en es fr it hi)\n"
"  -j, --jobs N         parallel TTS processes (default: CPU cores / 2)\n"
"  -w, --workdir DIR    intermediate files (default: <input>.work)\n"
"      --no-subs        do not embed translated subtitles\n"
"  -f, --force          redo every step instead of reusing work files\n"
"  -h, --help\n");
	exit(code);
}

int main(int argc, char **argv)
{
	static struct option lo[] = {
		{ "input", 1, 0, 'i' }, { "lang", 1, 0, 'l' }, { "output", 1, 0, 'o' },
		{ "audio", 1, 0, 'a' }, { "source", 1, 0, 's' }, { "model", 1, 0, 'm' },
		{ "prompt", 1, 0, 'p' }, { "voice", 1, 0, 'v' }, { "device", 1, 0, 'd' },
		{ "translator", 1, 0, 't' }, { "best", 0, 0, 'B' },
		{ "bg", 1, 0, 'b' }, { "jobs", 1, 0, 'j' }, { "workdir", 1, 0, 'w' },
		{ "no-subs", 0, 0, 'S' }, { "no-separate", 0, 0, 'N' },
		{ "lipsync", 0, 0, 'L' },
		{ "force", 0, 0, 'f' }, { "help", 0, 0, 'h' }, { 0 }
	};
	char *langs[MAX_LANGS], *langarg = NULL;
	int nlangs = 0, c;

	while ((c = getopt_long(argc, argv, "i:l:o:a:s:m:p:v:d:t:b:j:w:fh", lo, NULL)) != -1) {
		switch (c) {
		case 'i': opt.input = optarg; break;
		case 'l': langarg = optarg; break;
		case 'o': opt.output = optarg; break;
		case 'a': opt.audio = optarg; break;
		case 's': opt.source = optarg; break;
		case 'm': opt.model = optarg; break;
		case 'p': opt.prompt = optarg; break;
		case 'v': opt.voice = optarg; break;
		case 'd': opt.device = optarg; break;
		case 't': opt.translator = optarg; break;
		case 'B': opt.best = 1; break;
		case 'b': opt.bg = atof(optarg); break;
		case 'j': opt.jobs = atoi(optarg) > 0 ? atoi(optarg) : 1; break;
		case 'w': opt.workdir = optarg; break;
		case 'S': opt.subs = 0; break;
		case 'N': opt.separate = 0; break;
		case 'L': opt.lipsync = 1; break;
		case 'f': opt.force = 1; break;
		case 'h': usage(0); break;
		default:  usage(1);
		}
	}
	if (!opt.input || !langarg)
		usage(1);
	if (!file_ok(opt.input))
		die("cannot open %s", opt.input);
	for (char *save, *t = strtok_r(xstrdup(langarg), ",", &save); t;
	     t = strtok_r(NULL, ",", &save))
		if (*trim(t)) {
			if (nlangs == MAX_LANGS)
				die("too many languages");
			langs[nlangs++] = trim(t);
		}
	if (!nlangs)
		usage(1);
	if (!opt.jobs) {
		long cpus = sysconf(_SC_NPROCESSORS_ONLN);
		opt.jobs = cpus > 1 ? cpus / 2 : 1;
	}
	if (opt.bg < 0)
		opt.bg = opt.separate ? 1.0 : 0.0;
	if (strcmp(opt.translator, "auto") && strcmp(opt.translator, "llm") &&
	    strcmp(opt.translator, "nllb"))
		die("--translator must be auto, llm or nllb");
	setenv("DUBBER_TRANSLATOR", opt.translator, 1);
	setenv("DUBBER_PROMPT", opt.prompt ? opt.prompt : "", 1);
	if (opt.best) {
		static char big[PATH_MAX];
		snprintf(big, sizeof big,
			 "%s/.local/share/dubber/llm/google_gemma-3-12b-it-Q4_K_M.gguf",
			 home());
		if (!file_ok(big))
			die("--best needs %s\n        (DUBBER_LLM_SIZE=12b ./fetch_models.sh)", big);
		setenv("DUBBER_LLM_PIVOT", big, 1);
	}

	/* helpers live next to the executable */
	char exe[PATH_MAX];
	ssize_t l = readlink("/proc/self/exe", exe, sizeof exe - 1);
	if (l < 0)
		die("readlink /proc/self/exe failed");
	exe[l] = 0;
	snprintf(helper_dir, sizeof helper_dir, "%s", dirname(exe));

	/* base name for default output / workdir: input without extension */
	static char base[PATH_MAX], workbuf[PATH_MAX];
	snprintf(base, sizeof base, "%s", opt.input);
	char *dot = strrchr(base, '.');
	if (dot && !strchr(dot, '/'))
		*dot = 0;
	if (!opt.workdir) {
		snprintf(workbuf, sizeof workbuf, "%s.work", base);
		opt.workdir = workbuf;
	}
	if (mkdir(opt.workdir, 0755) < 0 && errno != EEXIST)
		die("cannot create %s: %s", opt.workdir, strerror(errno));

	setenv("PYTHONUNBUFFERED", "1", 1);   /* live helper progress in logs */
	pick_python();

	int need_piper = !opt.lipsync;
	for (int k = 0; k < nlangs; k++)
		if (opt.lipsync && !kokoro_lang(langs[k])) {
			fprintf(stderr, "dubber: no lip-sync voices for '%s'; it is "
				"dubbed the usual way (Piper)\n", langs[k]);
			need_piper = 1;
		}
	const char *tools[] = { "ffmpeg", "ffprobe", opt.python, "piper" };
	for (size_t i = 0; i < sizeof tools / sizeof *tools - !need_piper; i++)
		if (!in_path(tools[i]))
			die("'%s' not found in PATH", tools[i]);

	/* fail early, before the long GPU steps */
	char voice[MAX_LANGS][PATH_MAX], output[MAX_LANGS][PATH_MAX];
	for (int k = 0; k < nlangs; k++) {
		if (!opt.lipsync || !kokoro_lang(langs[k]))
			find_voice(langs[k], voice[k], sizeof voice[k]);
		if (opt.output && nlangs == 1) {
			snprintf(output[k], PATH_MAX, "%s", opt.output);
		} else if (opt.output) {
			char tmp[PATH_MAX];
			snprintf(tmp, sizeof tmp, "%s", opt.output);
			char *d = strrchr(tmp, '.');
			const char *ext = ".mp4";
			if (d && !strchr(d, '/')) {
				ext = opt.output + (d - tmp);
				*d = 0;
			}
			snprintf(output[k], PATH_MAX, "%s_%s%s", tmp, langs[k], ext);
		} else {
			snprintf(output[k], PATH_MAX, "%s_%s.mp4", base, langs[k]);
		}
	}

	const char *speech = opt.audio ? opt.audio : opt.input;
	if (!has_audio(speech))
		die("%s has no audio track. If it is a video-only download (e.g.\n"
		    "        YouTube DASH), get the matching audio stream and pass it with -a",
		    speech);

	double total = media_duration(opt.input);
	if (total <= 0)
		die("cannot read duration of %s", opt.input);

	pick_device();

	char source[PATH_MAX], vocals[PATH_MAX], background[PATH_MAX], srt[PATH_MAX];
	snprintf(source, sizeof source, "%s/source.wav", opt.workdir);
	snprintf(vocals, sizeof vocals, "%s/vocals.wav", opt.workdir);
	snprintf(background, sizeof background, "%s/background.wav", opt.workdir);

	extract_audio(speech, source);
	pid_t lpid = opt.lipsync ? lips_start() : 0;   /* CPU, beside the GPU */
	if (opt.separate)
		separate(source);
	const char *src = transcribe(opt.separate ? vocals : source, srt, sizeof srt);
	if (opt.lipsync) {
		if (!opt.separate)
			die("--lipsync needs the separated voice (drop --no-separate)");
		lips_finish(lpid);
		scene(srt, sizeof srt);
	}

	int n;
	seg_t *segs = parse_srt(srt, &n);
	if (!n)
		die("no speech recognized");
	fprintf(stderr, "    %d segments\n", n);

	char in[PATH_MAX];
	size_t len = 1;
	for (int i = 0; i < n; i++)
		len += strlen(segs[i].src) + 1;
	char *text = xmalloc(len), *p = text;
	for (int i = 0; i < n; i++)
		p += sprintf(p, "%s\n", segs[i].src);
	snprintf(in, sizeof in, "%s/src.txt", opt.workdir);
	if (!same_text(in, text)) {
		retranslate = 1;
		spit(in, text);
	}
	free(text);

	/* seconds each line may take: the translator keeps lines that short */
	char slots[PATH_MAX], *sl = xmalloc(n * 16 + 1);
	p = sl;
	for (int i = 0; i < n; i++)
		p += sprintf(p, "%.3f\n", slot_of(segs, n, i, total));
	snprintf(slots, sizeof slots, "%s/slots.txt", opt.workdir);
	if (!same_text(slots, sl)) {
		retranslate = 1;
		spit(slots, sl);
	}
	free(sl);

	/* other translator or context: old translations are stale */
	char tkey[PATH_MAX], tval[4096];
	snprintf(tkey, sizeof tkey, "%s/translate.key", opt.workdir);
	snprintf(tval, sizeof tval, "%s\n%s\n%d\n", opt.translator,
		 opt.prompt ? opt.prompt : "", opt.best);
	if (!same_text(tkey, tval)) {
		retranslate = 1;
		spit(tkey, tval);
	}

	/* GPU translates language k+1 while the CPU voices language k */
	pid_t tpid = translate_start(src, langs[0]);
	for (int k = 0; k < nlangs; k++) {
		const char *lang = langs[k];
		char dubsrt[PATH_MAX], dubwav[PATH_MAX];
		snprintf(dubsrt, sizeof dubsrt, "%s/%s.srt", opt.workdir, lang);
		snprintf(dubwav, sizeof dubwav, "%s/dub_%s.wav", opt.workdir, lang);

		int lips = opt.lipsync && kokoro_lang(lang);
		translate_finish(tpid, segs, n, src, lang);
		/* lipdub.py needs the GPU itself: no translation beside it */
		tpid = k + 1 < nlangs && !lips ? translate_start(src, langs[k + 1]) : 0;

		if (lips) {
			lipdub(segs, n, src, lang);
			write_srt(dubsrt, segs, n);
			place(segs, n, total, lang, dubwav);
			if (k + 1 < nlangs)
				tpid = translate_start(src, langs[k + 1]);
		} else {
			write_srt(dubsrt, segs, n);
			synthesize(segs, n, total, voice[k], lang, dubwav);
		}
		const char *bgsrc = !(opt.bg > 0) ? NULL
				  : opt.separate ? background : speech;
		mux(output[k], dubwav, dubsrt, bgsrc, opt.bg);
		step("done: %s  (subtitles: %s)", output[k], dubsrt);
	}
	return 0;
}
