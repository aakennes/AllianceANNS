// Test-only, single-thread AIO observer for the 4 KiB edge / 8 KiB vector case.
// Records native requests/completions without changing their arguments or data.
#include "support/io_observer.hpp"
#include <libaio.h>
#include <dlfcn.h>
#include <unistd.h>
#include <cstdio>
#include <cstring>
#include <stdexcept>
namespace {
bool active=false;
IOCounters counts;
char expected[4096]{};
uint64_t vector_base=0;
struct Pending {uint64_t key=0, bytes=0; bool occupied=false;};
Pending pending[256];
int cached_fd=-1; bool cached_valid=false;
bool ValidFD(int fd) {
  if(fd==cached_fd)return cached_valid;
  char name[64],path[4096];snprintf(name,sizeof(name),"/proc/self/fd/%d",fd);
  auto n=readlink(name,path,sizeof(path)-1);
  if(n<0)return false;path[n]=0;
  cached_fd=fd;cached_valid=strcmp(path,expected)==0;return cached_valid;
}
void Accepted(const iocb* cb) {
  ++counts.requests;counts.requested_bytes+=cb->u.c.nbytes;
  counts.unknown_fd+=!ValidFD(cb->aio_fildes);
  const bool vector=cb->u.c.offset>=vector_base;
  if(vector)++counts.vector_requests;else ++counts.graph_requests;
  const size_t bytes=vector?8192:4096;
  counts.bad_layout+=cb->aio_lio_opcode!=IO_CMD_PREAD || cb->u.c.nbytes!=bytes || cb->u.c.offset%bytes;
  const uint64_t key=reinterpret_cast<uint64_t>(cb->data);
  for(auto& p:pending)if(p.occupied && p.key==key){++counts.errors;return;}
  for(auto& p:pending)if(!p.occupied){p={key,cb->u.c.nbytes,true};++counts.pending;return;}
  ++counts.errors;
}
void Completed(const io_event& e) {
  ++counts.completions;
  const int64_t res=static_cast<int64_t>(e.res);
  if(res>0)counts.completed_bytes+=res;
  const uint64_t key=reinterpret_cast<uint64_t>(e.data);
  for(auto& p:pending)if(p.occupied && p.key==key){
    counts.errors+=res!=static_cast<int64_t>(p.bytes) || e.res2!=0;
    p.occupied=false;--counts.pending;return;
  }
  ++counts.errors;
}
}
void IOStart(const char* ml_path, uint64_t vector_offset) {
  vector_base=vector_offset;
  if(active || counts.pending)throw std::runtime_error("observer already active/pending");
  counts={};for(auto& p:pending)p={};
  if(strlen(ml_path)>=sizeof(expected))throw std::runtime_error("observer path too long");
  strcpy(expected,ml_path);cached_fd=-1;cached_valid=false;active=true;
}
IOCounters IOStop(){active=false;return counts;}
extern "C" int io_submit(io_context_t context,long n,iocb** cbs) {
  using Fn=int(*)(io_context_t,long,iocb**);
  static auto real=reinterpret_cast<Fn>(dlsym(RTLD_NEXT,"io_submit"));
  if(!real)_exit(126);
  int ret=real(context,n,cbs);
  if(active){if(ret<0 || ret!=n)++counts.errors;for(int i=0;i<ret;++i)Accepted(cbs[i]);}
  return ret;
}
extern "C" int io_getevents(io_context_t context,long min,long n,io_event* events,timespec* timeout) {
  using Fn=int(*)(io_context_t,long,long,io_event*,timespec*);
  static auto real=reinterpret_cast<Fn>(dlsym(RTLD_NEXT,"io_getevents"));
  if(!real)_exit(126);
  int ret=real(context,min,n,events,timeout);
  if(active){if(ret<0)++counts.errors;for(int i=0;i<ret;++i)Completed(events[i]);}
  return ret;
}
