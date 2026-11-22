// Compile-time syscall injection exercises the production IOHandler header
// without timing-dependent kernel queue exhaustion or real disk IO.
#include <libaio.h>
#include <cassert>
#include <cerrno>
#include <deque>
#include <iostream>
#include <vector>

namespace fake_aio {
struct Context {
  std::deque<io_event> events;
};
static std::vector<Context *> contexts;
static int next_submit = -999;
static int interrupted_polls = 0;
static int interrupted_destroys = 0;
static int again_polls = 0;
static int fail_setup_after = -1;
static int fatal_polls = 1;
static io_context_t fatal_poll = nullptr;
static size_t completions = 0, barriers = 0;
static size_t poll_calls = 0, blocking_poll_calls = 0;
static int empty_polls = 0;
int Setup(int, io_context_t *ctx) {
  if (fail_setup_after >= 0 && fail_setup_after-- == 0)
    return -ENOMEM;
  auto *c = new Context;
  contexts.push_back(c);
  *ctx = reinterpret_cast<io_context_t>(c);
  return 0;
}
int Destroy(io_context_t ctx) {
  if (interrupted_destroys) {
    --interrupted_destroys;
    return -EINTR;
  }
  auto *c = reinterpret_cast<Context *>(ctx);
  ++barriers;
  // Destroy's kernel contract guarantees all pending accesses have stopped.
  c->events.clear();
  return 0;
}
int Submit(io_context_t ctx, long n, iocb **cbs) {
  int accepted = next_submit == -999 ? static_cast<int>(n) : next_submit;
  next_submit = -999;
  auto *c = reinterpret_cast<Context *>(ctx);
  for (int i = 0; i < accepted; ++i) {
    io_event e{};
    e.data = cbs[i]->data;
    e.res = cbs[i]->u.c.nbytes;
    c->events.push_back(e);
  }
  return accepted;
}
int GetEvents(io_context_t ctx, long min_events, long max_events, io_event *out, timespec *) {
  ++poll_calls;
  blocking_poll_calls += min_events > 0;
  if (empty_polls) {
    --empty_polls;
    return 0;
  }
  if (interrupted_polls) {
    --interrupted_polls;
    return -EINTR;
  }
  if (again_polls) {
    --again_polls;
    return -EAGAIN;
  }
  if (ctx == fatal_poll) {
    if (--fatal_polls == 0)
      fatal_poll = nullptr;
    return -EIO;
  }
  auto *c = reinterpret_cast<Context *>(ctx);
  assert(min_events == 0 || !c->events.empty());
  // One event at a time also checks that an unrelated older completion does
  // not accidentally satisfy the accepted-prefix drain condition.
  if (c->events.empty() || max_events == 0)
    return 0;
  out[0] = c->events.front();
  c->events.pop_front();
  ++completions;
  return 1;
}
void CheckEmpty() {
  for (auto *c : contexts)
    assert(c->events.empty());
}
} // namespace fake_aio

#define io_setup fake_aio::Setup
#define io_destroy fake_aio::Destroy
#define io_submit fake_aio::Submit
#define io_getevents fake_aio::GetEvents
#include "io/io_handler.hpp"
#undef io_setup
#undef io_destroy
#undef io_submit
#undef io_getevents

int main() {
  using namespace alliance;
  using namespace fake_aio;
  IOQueueObserver observer;
  {
    IOHandler io(8);
    io.ResetQueueProfile(&observer);
    IORequest old_edge, old_vec;
    old_edge.len = old_vec.len = 4096;
    io.SubmitRead(old_edge);
    io.SubmitVecRead(old_vec);
    {
      std::vector<IORequest> reqs(4);
      std::vector<IORequest *> batch;
      for (auto &r : reqs) {
        r.len = 4096;
        batch.push_back(&r);
      }
      next_submit = 2;
      interrupted_polls = 1;
      bool threw = false;
      try {
        io.SubmitVecReadBatch(batch);
      } catch (const std::runtime_error &) {
        threw = true;
      }
      assert(threw);
      assert(old_edge.finished && old_vec.finished);
      assert(reqs[0].finished && reqs[1].finished);
      assert(reqs[0].res == 4096 && reqs[1].res == 4096);
      assert(!reqs[2].finished && !reqs[3].finished);
      assert(completions == 4);
      CheckEmpty(); // No event retains a pointer when reqs leaves scope.
    }
    assert(observer.current == 0 && io.QueueProfile().Pending() == 0);
    io.PollAll();
    io.PollVecAll();
    CheckEmpty();
    // A recoverable partial submission does not poison either context.
    IORequest next;
    next.len = 4096;
    io.SubmitRead(next);
    io.PollWait();
    assert(next.finished && next.res == 4096);

    // Negative/zero returns accept nothing, but earlier pending work must
    // still drain before the new exception unwinds the caller's stack.
    for (const int failed_submit : {-EAGAIN, 0}) {
      IORequest prior, rejected;
      prior.len = rejected.len = 4096;
      io.SubmitVecRead(prior);
      next_submit = failed_submit;
      bool threw = false;
      try {
        io.SubmitWriteBatch({&rejected});
      } catch (const std::runtime_error &) {
        threw = true;
      }
      assert(threw && prior.finished && !rejected.finished);
      CheckEmpty();
    }

    // The write batch path has the same accepted-prefix lifetime contract.
    IORequest a, b;
    a.len = b.len = 4096;
    next_submit = 1;
    bool threw = false;
    try {
      io.SubmitWriteBatch({&a, &b});
    } catch (const std::runtime_error &) {
      threw = true;
    }
    assert(threw && a.finished && !b.finished);
    CheckEmpty();
  }
  {
    IOHandler io(8);
    io.ResetQueueProfile(&observer);
    IORequest old_edge, old_vec;
    old_edge.len = old_vec.len = 4096;
    io.SubmitRead(old_edge);
    io.SubmitVecRead(old_vec);
    IORequest a, b;
    a.len = b.len = 4096;
    fatal_poll = reinterpret_cast<io_context_t>(contexts.back());
    const size_t before = barriers;
    next_submit = 1;
    interrupted_destroys = 1;
    bool threw = false;
    try {
      io.SubmitVecReadBatch({&a, &b});
    } catch (const std::runtime_error &) {
      threw = true;
    }
    assert(threw && barriers == before + 1);
    assert(old_edge.finished && interrupted_destroys == 0);
    assert(a.finished && a.res == -ECANCELED && !b.finished);
    CheckEmpty();
    threw = false;
    try {
      io.SubmitRead(b);
    } catch (const std::runtime_error &) {
      threw = true;
    }
    assert(threw); // A destroyed context cannot silently be reused.
    assert(observer.current == 0 && io.QueueProfile().Pending() == 0);
  }
  CheckEmpty();
  assert(observer.current == 0);
  // Constructor failure and normal destruction must retry interrupted barriers.
  {
    const auto before = barriers;
    fail_setup_after = 1;
    interrupted_destroys = 2;
    bool threw = false;
    try {
      IOHandler io;
    } catch (const std::runtime_error &) {
      threw = true;
    }
    assert(threw && barriers == before + 1 && interrupted_destroys == 0);
    fail_setup_after = -1;
  }
  {
    IOHandler io;
    interrupted_destroys = 2;
  }
  assert(interrupted_destroys == 0);
  // The guard precedes request/buffer destruction, including ordinary C++ errors.
  {
    IOHandler io;
    io.ResetQueueProfile(&observer);
    struct Storage {
      IORequest edge, vec;
      ~Storage() {
        CheckEmpty();
      }
    };
    for (int repeat = 0; repeat < 3; ++repeat) {
      try {
        Storage storage;
        IOCompletionGuard completion(io);
        storage.edge.len = storage.vec.len = 4096;
        io.SubmitRead(storage.edge);
        io.SubmitVecRead(storage.vec);
        throw std::bad_alloc();
      } catch (const std::bad_alloc &) {
      }
      CheckEmpty();
      assert(observer.current == 0);
    }
    // Reopening cannot carry outstanding accesses over to the new file.
    IORequest request;
    request.len = 4096;
    io.SubmitRead(request);
    bool threw = false;
    try {
      io.Open("/proc/self/no-such-r5b-file");
    } catch (const std::runtime_error &) {
      threw = true;
    }
    assert(threw && request.finished);
    CheckEmpty();
  }
  // Every public poll entry surfaces permanent errors after draining BOTH queues.
  for (int poll = 0; poll < 5; ++poll) {
    IOHandler io;
    io.ResetQueueProfile(&observer);
    IORequest edge, vec;
    edge.len = vec.len = 4096;
    IOCompletionGuard completion(io);
    io.SubmitRead(edge);
    io.SubmitVecRead(vec);
    fatal_poll = reinterpret_cast<io_context_t>(contexts[contexts.size() - (poll < 3 ? 2 : 1)]);
    fatal_polls = 1;
    bool threw = false;
    try {
      switch (poll) {
      case 0:
        io.PollAll();
        break;
      case 1:
        io.PollWait();
        break;
      case 2:
        io.PollWaitBatch();
        break;
      case 3:
        io.PollVecAll();
        break;
      case 4:
        io.PollVecWait();
        break;
      }
    } catch (const std::runtime_error &) {
      threw = true;
    }
    assert(threw && edge.finished && vec.finished && observer.current == 0);
    CheckEmpty();
    // A subsequently healthy queue remains reusable after drained errors.
    IORequest next;
    next.len = 4096;
    io.SubmitRead(next);
    interrupted_polls = 1;
    again_polls = 1;
    io.PollWait();
    assert(next.finished && !interrupted_polls && !again_polls);
  }
  {
    IOHandler io;
    io.ResetQueueProfile(&observer);
    IORequest edge, vec;
    edge.len = vec.len = 4096;
    IOCompletionGuard completion(io);
    io.SubmitRead(edge);
    io.SubmitVecRead(vec);
    interrupted_polls = 1;
    io.PollAll();
    assert(!edge.finished);
    again_polls = 1;
    io.PollVecAll();
    assert(!vec.finished);
    fatal_poll = reinterpret_cast<io_context_t>(contexts.back());
    fatal_polls = 2;
    interrupted_destroys = 1;
    const auto before = barriers;
    bool threw = false;
    try {
      io.PollVecWait();
    } catch (const std::runtime_error &) {
      threw = true;
    }
    assert(threw && barriers == before + 1 && observer.current == 0);
    CheckEmpty();
    threw = false;
    try {
      io.SubmitRead(edge);
    } catch (const std::runtime_error &) {
      threw = true;
    }
    assert(threw); // Destruction fallback poisons the handler.
  }
  CheckEmpty();

  // Busy waits must tolerate an initially empty completion queue without
  // blocking in the kernel; ordinary PollAll still returns after one poll.
  {
    IOHandler io(8, IOHandler::CompletionWait::BusyPoll);
    io.ResetQueueProfile(&observer);
    const auto blocking_before = blocking_poll_calls;
    IORequest edge, vec;
    edge.len = vec.len = 4096;
    io.SubmitRead(edge);
    empty_polls = 3;
    const auto before = poll_calls;
    io.PollAll();
    assert(poll_calls == before + 1 && !edge.finished && empty_polls == 2);
    io.PollWaitBatch();
    assert(edge.finished && edge.res == 4096 && empty_polls == 0);
    io.SubmitVecRead(vec);
    empty_polls = 2;
    interrupted_polls = 1;
    again_polls = 1;
    io.PollVecWait();
    assert(vec.finished && vec.res == 4096 && empty_polls == 0);
    assert(interrupted_polls == 0 && again_polls == 0);
    assert(blocking_poll_calls == blocking_before);
    bool threw = false;
    try {
      io.PollWaitBatch(0);
    } catch (const std::runtime_error &) {
      threw = true;
    }
    assert(threw); // An invalid width must not become an endless spin.
    CheckEmpty();
    assert(observer.current == 0);
  }
  // Accepted-prefix recovery must also finish delayed requests in busy mode.
  {
    IOHandler io(8, IOHandler::CompletionWait::BusyPoll);
    io.ResetQueueProfile(&observer);
    const auto blocking_before = blocking_poll_calls;
    IORequest edge, a, b;
    edge.len = a.len = b.len = 4096;
    io.SubmitRead(edge);
    next_submit = 1;
    empty_polls = 3;
    bool threw = false;
    try {
      io.SubmitVecReadBatch({&a, &b});
    } catch (const std::runtime_error &) {
      threw = true;
    }
    assert(threw && edge.finished && a.finished && !b.finished);
    assert(empty_polls == 0 && blocking_poll_calls == blocking_before);
    CheckEmpty();
    assert(observer.current == 0);
  }
  // Permanent errors still drain both queues and preserve request lifetimes.
  {
    IOHandler io(8, IOHandler::CompletionWait::BusyPoll);
    io.ResetQueueProfile(&observer);
    const auto blocking_before = blocking_poll_calls;
    IORequest edge, vec;
    edge.len = vec.len = 4096;
    io.SubmitRead(edge);
    io.SubmitVecRead(vec);
    fatal_poll = reinterpret_cast<io_context_t>(contexts[contexts.size() - 2]);
    fatal_polls = 1;
    bool threw = false;
    try {
      io.PollWait();
    } catch (const std::runtime_error &) {
      threw = true;
    }
    assert(threw && edge.finished && vec.finished);
    assert(blocking_poll_calls == blocking_before);
    CheckEmpty();
    assert(observer.current == 0);
  }

  for (auto *c : contexts)
    delete c;
  std::cout << "IOHandler submit/poll/unwind/barrier lifetime tests passed\n";
}
