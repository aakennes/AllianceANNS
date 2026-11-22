"""Synthetic subprocess/fault tests: never invoke a dataset search or systemd."""
import copy
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from benchlib import runner, scope_entry


class RunnerTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(dir=os.environ.get("TEST_TMPDIR"))
        self.root = Path(self.temporary.name).resolve()
        self.job = self.root / "job"
        self.fixture = self.root / "fixture.py"
        self.fixture.write_text(
            "import json,sys\nfrom pathlib import Path\n"
            "counter=Path(sys.argv[2])\n"
            "with counter.open('a') as f: f.write('spawn\\n')\n"
            "out=Path(sys.argv[1]);out.mkdir(parents=True,exist_ok=True)\n"
            "(out/'rows.json').write_text(json.dumps([{'repeat':i,'ef':16,'validated':True} for i in range(1,6)]))\n",
            encoding="utf-8")
        self.data = self.root / "data.bin"
        self.data.write_bytes(b"synthetic independent input\n")
        self.source = self.root / "source.py"
        self.source.write_text("synthetic executed source identity\n", encoding="utf-8")
        self.counter = self.root / "launches.txt"
        python = str(Path(sys.executable).resolve())
        self.spec = {
            "schema_version": 2, "id": "synthetic-runner", "memory": {
                "loading_limit_bytes": 9 << 30, "search_rss_limit_bytes": 3 << 30},
            "aggregation": {"policy": "qps-paired-trim", "repeats": 5},
            "inputs": [{"path": p, "sha256": runner._sha(p)}
                       for p in (python, str(self.data), str(self.fixture))],
            "lock_paths": [str(self.root / "shared.lock")],
            "curves": [{"id": "aa-origin", "algorithm": "allianceann", "workload": "origin",
                        "cache_policy": "landing", "efs": [16], "repeats": 5, "queries": 2,
                        "k": 1, "bits": 1, "binary_path": python,
                        "argv": [python, str(self.fixture), "{output}", str(self.counter)],
                        "env": {"OMP_NUM_THREADS": "1", "CASELAW_OUTPUT": "{output}"},
                        "scope_prefix": "caselaw-rss4-r12-", "expected": {},
                        "inputs": {key: str(self.data) for key in
                                   ("base", "query", "ranges", "gt", "attributes")}}]}
        self.patches = [mock.patch.object(runner, "_source_fingerprints", side_effect=lambda: {"fixture": runner._sha(self.source)}),
                        mock.patch.object(runner, "_finalize", side_effect=lambda spec, job, rows: {"rows": len(rows)}),
                        mock.patch.object(runner.tempfile, "gettempdir", return_value=str(self.root))]
        for patch in self.patches:
            patch.start()

    def tearDown(self):
        for patch in reversed(self.patches):
            patch.stop()
        self.temporary.cleanup()

    def count(self):
        return len(self.counter.read_text().splitlines()) if self.counter.exists() else 0

    def evidence(self, spec, curve, directory, exit_code=0):
        # Synthetic evidence exercises the runner contract only. Production
        # scope_entry independently reads these fields from the real kernel.
        sent = runner._read(directory / "sent.json")
        unit = curve["scope_prefix"] + sent["run_token"][:20] + ".scope"
        memory = {"schema_version": 1, "status": "finished", "exit_code": exit_code,
                  "request_sha256": sent["request_sha256"], "curve_id": curve["id"],
                  "run_token": sent["run_token"], "scope_unit": unit,
                  "cgroup_path": "/sys/fs/cgroup/test/" + unit,
                  "initial_memory_max": str(spec["memory"]["loading_limit_bytes"]),
                  "initial_memory_swap_max": "0", "final_memory_max": "max",
                  "final_memory_swap_max": "0", "affinity_before": [1], "affinity_after": [1],
                  "memory_events_before": {"oom": 0, "oom_kill": 0},
                  "memory_events_after": {"oom": 0, "oom_kill": 0}}
        runner._write(directory / "memory.json", memory)

    def execute(self, spec, curve, directory):
        self.assertTrue((directory / "sent.json").is_file())
        argv = [arg.replace("{output}", str(directory / "raw")) for arg in curve["argv"]]
        with (directory / "stdout.log").open("wb") as out, (directory / "stderr.log").open("wb") as err:
            result = subprocess.run(argv, stdout=out, stderr=err)
        self.evidence(spec, curve, directory, result.returncode)
        return {"exit_code": result.returncode}

    def verify(self, spec, curve, directory):
        return {"status": "verified", "rows": runner._read(directory / "raw" / "rows.json")}

    def run_job(self, **kwargs):
        kwargs.setdefault("executor", self.execute)
        kwargs.setdefault("verifier", self.verify)
        return runner.run_experiment(self.spec, self.job, **kwargs)

    def test_launch_once_and_reverify_on_resume(self):
        self.assertEqual(self.run_job()["status"], "verified")
        self.assertEqual(self.count(), 1)
        with self.assertRaisesRegex(runner.RunError, "requires --resume"):
            self.run_job()
        result = self.run_job(resume=True, executor=lambda *args: self.fail("relaunch"))
        self.assertEqual(result["row_count"], 5)
        proof = runner._read(self.job / "curves/aa-origin/proof.json")
        self.assertTrue(all(row["curve_id"] == "aa-origin" for row in proof["verification"]["rows"]))
        self.assertEqual(self.count(), 1)

    def test_sent_and_all_ancestors_are_synced_before_executor(self):
        synchronized = []
        sync = runner._sync_directory
        curve_dir = self.job / "curves/aa-origin"

        def record(path):
            sync(path)
            synchronized.append((Path(path).resolve(), (curve_dir / "sent.json").exists()))

        def execute(spec, curve, directory):
            # The post-write directory barrier must happen after sent.json
            # exists, and every link reaching this curve must be durable.
            self.assertIn((curve_dir, True), synchronized)
            reached = {path for path, _ in synchronized}
            self.assertTrue(set((curve_dir / "raw", curve_dir) + tuple(curve_dir.parents)) <= reached)
            return self.execute(spec, curve, directory)

        with mock.patch.object(runner, "_sync_directory", side_effect=record):
            self.run_job(executor=execute)
        self.assertEqual(self.count(), 1)

    def test_failed_sent_directory_sync_prevents_launch_and_resume_replay(self):
        sync = runner._sync_directory
        curve_dir = self.job / "curves/aa-origin"

        def fail(path):
            if Path(path) == curve_dir and (curve_dir / "sent.json").exists():
                raise OSError("synthetic directory fsync failure")
            sync(path)

        with mock.patch.object(runner, "_sync_directory", side_effect=fail):
            with self.assertRaisesRegex(OSError, "directory fsync failure"):
                self.run_job(executor=lambda *args: self.fail("launched without durable sent marker"))
        self.assertEqual(self.count(), 0)
        self.assertTrue((curve_dir / "sent.json").is_file())
        with self.assertRaisesRegex(runner.RunError, "Missing real scope"):
            self.run_job(resume=True, executor=lambda *args: self.fail("replayed uncertain marker"))
        self.assertEqual(self.count(), 0)

    @unittest.skipIf(os.name == "nt", "directory fsync is a POSIX production contract")
    def test_directory_sync_errors_are_not_suppressed(self):
        with mock.patch.object(runner.os, "fsync", side_effect=OSError("synthetic fsync failure")):
            with self.assertRaisesRegex(OSError, "fsync failure"):
                runner._sync_directory(self.root)

    def test_partial_sent_curve_never_relaunches(self):
        def partial(spec, curve, directory):
            self.execute(spec, curve, directory)
            (directory / "memory.json").unlink()
            raise RuntimeError("synthetic interrupt after spawn")
        with self.assertRaisesRegex(RuntimeError, "synthetic interrupt"):
            self.run_job(executor=partial)
        with self.assertRaisesRegex(runner.RunError, "Missing real scope"):
            self.run_job(resume=True, executor=lambda *args: self.fail("partial replay"))
        self.assertEqual(self.count(), 1)
        self.assertTrue((self.job / "fingerprints.after.json").is_file())

    def test_missing_finished_receipt_can_recover_real_scope_success(self):
        def interrupted(spec, curve, directory):
            self.execute(spec, curve, directory)
            raise RuntimeError("receipt interrupted")
        with self.assertRaises(RuntimeError):
            self.run_job(executor=interrupted)
        self.run_job(resume=True, executor=lambda *args: self.fail("replay"))
        path = self.job / "curves/aa-origin"
        self.assertFalse((path / "finished.json").exists())
        self.assertTrue(runner._read(path / "proof.json")["recovered_without_launcher_receipt"])
        self.assertEqual(self.count(), 1)

    def test_verification_failure_resumes_without_spawn_and_can_continue_unsent(self):
        second = copy.deepcopy(self.spec["curves"][0])
        second.update(id="aa-p1", workload="p1")
        self.spec["curves"].append(second)
        with self.assertRaisesRegex(runner.RunError, "verification failed"):
            self.run_job(verifier=lambda *args: {"status": "failed", "rows": []})
        self.assertEqual(self.count(), 1)
        self.assertFalse((self.job / "curves/aa-p1/sent.json").exists())
        self.run_job(resume=True)
        self.assertEqual(self.count(), 2)

    def test_interrupted_proof_commit_recovers_without_spawn(self):
        write = runner._write
        def interrupt(path, value, **kwargs):
            if Path(path).name == "state.json" and value.get("status") == "verified":
                raise RuntimeError("synthetic interrupted state commit")
            return write(path, value, **kwargs)
        with mock.patch.object(runner, "_write", side_effect=interrupt):
            with self.assertRaisesRegex(RuntimeError, "interrupted state commit"):
                self.run_job()
        self.assertTrue((self.job / "curves/aa-origin/proof.json").is_file())
        self.run_job(resume=True, executor=lambda *args: self.fail("replay"))
        self.assertEqual(self.count(), 1)

    def test_lock_contention_refuses_before_spawn(self):
        with runner._locks(self.spec["lock_paths"]):
            with self.assertRaisesRegex(runner.RunError, "Lock is held"):
                self.run_job()
        self.assertEqual(self.count(), 0)
        self.assertFalse((self.job / "request.json").exists())

    def test_existing_unlocked_file_is_reusable_and_never_unlinked(self):
        path = Path(self.spec["lock_paths"][0])
        path.write_text("old owner metadata", encoding="utf-8")
        inode = path.stat().st_ino
        with runner._locks([path]):
            self.assertEqual(path.stat().st_ino, inode)
        self.assertTrue(path.exists())
        self.run_job()
        self.assertEqual(path.stat().st_ino, inode)
        self.assertEqual(self.count(), 1)

    @unittest.skipIf(os.name == "nt", "production flock interoperability requires POSIX")
    def test_external_flock_and_runner_are_mutually_exclusive(self):
        path = self.spec["lock_paths"][0]
        hold = ("import fcntl,sys\n"
                "f=open(sys.argv[1],'a+')\nfcntl.flock(f,fcntl.LOCK_EX)\n"
                "print('locked',flush=True)\nsys.stdin.read()\n")
        process = subprocess.Popen([sys.executable, "-c", hold, path], stdin=subprocess.PIPE,
                                   stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)
        try:
            self.assertEqual(process.stdout.readline().strip(), "locked")
            with self.assertRaisesRegex(runner.RunError, "Lock is held"):
                self.run_job()
        finally:
            process.communicate(input="", timeout=10)
        attempt = ("import fcntl,sys\nf=open(sys.argv[1],'a+')\n"
                   "try: fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)\n"
                   "except BlockingIOError: sys.exit(3)\n")
        with runner._locks([path]):
            result = subprocess.run([sys.executable, "-c", attempt, path])
            self.assertEqual(result.returncode, 3)
        self.assertEqual(subprocess.run([sys.executable, "-c", attempt, path]).returncode, 0)
        self.assertEqual(self.count(), 0)

    def test_immutable_request_and_input_and_source_identity(self):
        self.run_job()
        changed = copy.deepcopy(self.spec)
        changed["curves"][0]["efs"] = [20]
        with self.assertRaisesRegex(runner.RunError, "request identity changed"):
            runner.run_experiment(changed, self.job, resume=True, executor=self.execute, verifier=self.verify)
        self.data.write_bytes(b"changed data\n")
        with self.assertRaisesRegex(ValueError, "SHA256 mismatch"):
            self.run_job(resume=True)
        self.data.write_bytes(b"synthetic independent input\n")
        # Restoring bytes alone does not restore the frozen stat identity.
        with self.assertRaisesRegex(runner.RunError, "identity changed"):
            self.run_job(resume=True)
        self.assertEqual(self.count(), 1)

    def test_source_change_rejected(self):
        self.run_job()
        self.source.write_text("changed runner source\n", encoding="utf-8")
        with self.assertRaisesRegex(runner.RunError, "identity changed"):
            self.run_job(resume=True)
        self.assertEqual(self.count(), 1)

    def test_verified_raw_tampering_or_missing_proof_refused(self):
        self.run_job()
        path = self.job / "curves/aa-origin"
        with (path / "raw/rows.json").open("a") as stream:
            stream.write(" ")
        with self.assertRaisesRegex(runner.RunError, "artifacts changed"):
            self.run_job(resume=True)
        (path / "proof.json").unlink()
        with self.assertRaisesRegex(runner.RunError, "no matching proof"):
            self.run_job(resume=True)
        self.assertEqual(self.count(), 1)

    def test_bare_success_and_nonzero_exit_do_not_pass(self):
        def failed(spec, curve, directory):
            self.execute(spec, curve, directory)
            return {"exit_code": 7}
        with self.assertRaisesRegex(runner.RunError, "launcher failed"):
            self.run_job(executor=failed)
        runner._write(self.job / "curves/aa-origin/state.json", {"status": "success"})
        with self.assertRaisesRegex(runner.RunError, "launcher failed"):
            self.run_job(resume=True)
        self.assertEqual(self.count(), 1)

    def test_environment_and_default_launch_contract(self):
        with mock.patch.dict(os.environ, {"LD_PRELOAD": "bad", "CASELAW_OLD": "bad",
                                         "PIPEWOW_FLAGS": "bad", "TCMALLOC_X": "bad",
                                         "MALLOC_CONF": "bad", "STABILITY_SEED": "bad",
                                         "XDG_RUNTIME_DIR": "/run/user/other-user",
                                         "DBUS_SESSION_BUS_ADDRESS": "unix:path=/run/user/other-user/bus"}):
            env = runner.clean_environment({"OMP_NUM_THREADS": "1"})
        for key in ("LD_PRELOAD", "CASELAW_OLD", "PIPEWOW_FLAGS", "TCMALLOC_X", "MALLOC_CONF", "STABILITY_SEED"):
            self.assertNotIn(key, env)
        self.assertEqual(env["OMP_NUM_THREADS"], "1")
        if os.name != "nt":
            self.assertEqual(env["XDG_RUNTIME_DIR"], "/run/user/" + str(os.getuid()))
            self.assertEqual(env["DBUS_SESSION_BUS_ADDRESS"], "unix:path=" + env["XDG_RUNTIME_DIR"] + "/bus")
        curve = self.spec["curves"][0]
        directory = self.root / "launch-contract"
        directory.mkdir()
        runner._write(directory / "sent.json", {"run_token": "f" * 32, "request_sha256": "a" * 64})
        with mock.patch.object(runner.subprocess, "run", return_value=mock.Mock(returncode=0)) as execute:
            runner.execute_curve(self.spec, curve, directory)
        argv = execute.call_args[0][0]
        self.assertEqual(argv[:4], ["systemd-run", "--user", "--scope", "--quiet"])
        self.assertIn("MemoryMax=" + str(9 << 30), argv)
        self.assertIn("MemorySwapMax=0", argv)
        self.assertEqual(argv[argv.index("taskset") + 1:argv.index("taskset") + 3], ["-c", "1"])
        self.assertEqual(execute.call_args[1]["env"]["CASELAW_OUTPUT"], str(directory / "raw"))

    def test_all_default_curves_preflight_before_any_spawn(self):
        from benchlib import validation
        second = copy.deepcopy(self.spec["curves"][0])
        second.update(id="aa-p1", workload="p1")
        self.spec["curves"].append(second)
        calls = []
        def preflight(spec, curve):
            calls.append(curve["id"])
            if curve["id"] == "aa-p1":
                raise RuntimeError("synthetic malformed second input")
            return {"status": "ready"}
        with mock.patch.object(validation, "preflight_curve", side_effect=preflight), \
                mock.patch.object(runner, "execute_curve", side_effect=self.execute):
            with self.assertRaisesRegex(RuntimeError, "malformed second input"):
                runner.run_experiment(self.spec, self.job)
        self.assertEqual(calls, ["aa-origin", "aa-p1"])
        self.assertEqual(self.count(), 0)
        self.assertFalse(list(self.job.glob("curves/*/sent.json")))

    def test_scope_requires_real_phase_release_and_oom_counters(self):
        cgroup = self.root / "caselaw-rss4-r12-fixture.scope"
        cgroup.mkdir()
        for name, value in {"memory.max": str(9 << 30), "memory.swap.max": "0",
                            "memory.events": "oom 0\noom_kill 0\n"}.items():
            (cgroup / name).write_text(value, encoding="utf-8")
        config = {"scope_unit": cgroup.name, "request_sha256": "a" * 64, "curve_id": "aa-origin",
                  "run_token": "x", "memory_path": str(self.root / "memory.json"),
                  "loading_limit_bytes": 9 << 30, "argv": ["never-a-native-command"], "cwd": str(self.root)}
        def native(*args, **kwargs):
            (cgroup / "memory.max").write_text("max", encoding="utf-8")
            return 0
        child = mock.Mock(pid=123, returncode=0)
        child.wait.side_effect = native
        child.poll.return_value = 0
        with mock.patch.object(scope_entry, "_cgroup", return_value=cgroup), \
                mock.patch.object(scope_entry.os, "sched_getaffinity", return_value={1}, create=True), \
                mock.patch.object(scope_entry.subprocess, "Popen", return_value=child):
            self.assertEqual(scope_entry.run(config), 0)
        memory = runner._read(config["memory_path"])
        self.assertEqual(memory["final_memory_max"], "max")
        self.assertNotIn("loading_peak_rss_bytes", memory)
        (cgroup / "memory.max").write_text(str(9 << 30), encoding="utf-8")
        child.wait.side_effect = None
        child.wait.return_value = 0
        with mock.patch.object(scope_entry, "_cgroup", return_value=cgroup), \
                mock.patch.object(scope_entry.os, "sched_getaffinity", return_value={1}, create=True), \
                mock.patch.object(scope_entry.subprocess, "Popen", return_value=child):
            self.assertEqual(scope_entry.run(config), 125)
        self.assertEqual(runner._read(config["memory_path"])["status"], "failed")
        child.poll.return_value = None
        child.returncode = -15
        child.wait.side_effect = [KeyboardInterrupt(), -15]
        with mock.patch.object(scope_entry, "_cgroup", return_value=cgroup), \
                mock.patch.object(scope_entry.os, "sched_getaffinity", return_value={1}, create=True), \
                mock.patch.object(scope_entry.subprocess, "Popen", return_value=child):
            self.assertEqual(scope_entry.run(config), 125)
        interrupted = runner._read(config["memory_path"])
        self.assertEqual(interrupted["child_cleanup"]["status"], "reaped")
        child.terminate.assert_called_once_with()

    def test_executor_interrupt_stops_only_its_scope_without_relaunch(self):
        curve = self.spec["curves"][0]
        directory = self.root / "interrupt"
        directory.mkdir()
        token = "f" * 32
        unit = curve["scope_prefix"] + token[:20] + ".scope"
        runner._write(directory / "sent.json", {"run_token": token, "request_sha256": "a" * 64})
        ok = mock.Mock(returncode=0, stdout="", stderr="")
        stopped = mock.Mock(returncode=0, stdout="ActiveState=inactive\nLoadState=loaded\n", stderr="")
        with mock.patch.object(runner.subprocess, "run", side_effect=[KeyboardInterrupt(), ok, stopped]) as run:
            with self.assertRaises(KeyboardInterrupt):
                runner.execute_curve(self.spec, curve, directory)
        calls = run.call_args_list
        self.assertEqual(sum(call[0][0][0] == "systemd-run" for call in calls), 1)
        self.assertEqual(calls[1][0][0], ["systemctl", "--user", "stop", unit])
        self.assertTrue(all(call[0][0][-1] == unit for call in calls[1:]))
        self.assertEqual(runner._read(directory / "cleanup.json")["status"], "stopped")

    def test_scope_cleanup_fallback_is_bounded_and_failure_is_explicit(self):
        unit = "caselaw-rss4-r12-only-this-launch.scope"
        active = mock.Mock(returncode=0, stdout="ActiveState=active\nLoadState=loaded\n", stderr="")
        inactive = mock.Mock(returncode=0, stdout="ActiveState=inactive\nLoadState=loaded\n", stderr="")
        ok = mock.Mock(returncode=0, stdout="", stderr="")
        with mock.patch.object(runner.subprocess, "run", side_effect=[ok, active, ok, ok, inactive]) as run:
            self.assertEqual(runner._stop_scope(unit, {}, self.root)["status"], "stopped")
        self.assertIn(["systemctl", "--user", "kill", "--kill-who=all", "--signal=KILL", unit],
                      [call[0][0] for call in run.call_args_list])
        with mock.patch.object(runner.subprocess, "run", side_effect=OSError("synthetic unavailable bus")) as run:
            with self.assertRaisesRegex(runner.RunError, "Could not confirm"):
                runner._stop_scope(unit, {}, self.root)
        self.assertEqual(run.call_count, 5)
        self.assertTrue(all(0 < call[1]["timeout"] <= 10 for call in run.call_args_list))
        self.assertEqual(runner._read(self.root / "cleanup.json")["status"], "failed")

    def test_scope_child_interruption_terminates_then_kills_and_reaps(self):
        child = mock.Mock(pid=456, returncode=-9)
        child.poll.return_value = None
        child.wait.side_effect = [subprocess.TimeoutExpired("synthetic", 5), -9]
        cleanup = scope_entry._terminate_child(child)
        self.assertEqual(cleanup["status"], "reaped")
        self.assertEqual(cleanup["actions"], ["terminate", "kill"])
        child.terminate.assert_called_once_with()
        child.kill.assert_called_once_with()
        self.assertEqual(child.wait.call_args_list, [mock.call(timeout=5), mock.call(timeout=5)])
        child.wait.side_effect = subprocess.TimeoutExpired("synthetic", 5)
        self.assertEqual(scope_entry._terminate_child(child)["status"], "failed")


if __name__ == "__main__":
    unittest.main()
