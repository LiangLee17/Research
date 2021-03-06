/*
    Copyright 2005-2013 Intel Corporation.  All Rights Reserved.

    This file is part of Threading Building Blocks.

    Threading Building Blocks is free software; you can redistribute it
    and/or modify it under the terms of the GNU General Public License
    version 2 as published by the Free Software Foundation.

    Threading Building Blocks is distributed in the hope that it will be
    useful, but WITHOUT ANY WARRANTY; without even the implied warranty
    of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with Threading Building Blocks; if not, write to the Free Software
    Foundation, Inc., 51 Franklin St, Fifth Floor, Boston, MA  02110-1301  USA

    As a special exception, you may use this file as part of a free software
    library without restriction.  Specifically, if other files instantiate
    templates or use macros or inline functions from this file, or you compile
    this file and link it with other files to produce an executable, this
    file does not by itself cause the resulting executable to be covered by
    the GNU General Public License.  This exception does not however
    invalidate any other reasons why the executable file might be covered by
    the GNU General Public License.
*/

#if __TBB_CPF_BUILD
    #define TBB_PREVIEW_TASK_ARENA 1
#endif
#include "harness.h"


#if __TBB_TASK_GROUP_CONTEXT

#include "tbb/task.h"
#include "tbb/task_scheduler_init.h"
#include "tbb/atomic.h"
#include <cstdlib>

#if _MSC_VER && __TBB_NO_IMPLICIT_LINKAGE
// plays around __TBB_NO_IMPLICIT_LINKAGE. __TBB_LIB_NAME should be defined (in makefiles)
    #pragma comment(lib, __TBB_STRING(__TBB_LIB_NAME))
#endif

const int NumIterations = 100;
const int NumLeafTasks = 2;
int MinBaseDepth = 9;
int MaxBaseDepth = 11;
int BaseDepth = 0;

const int NumTests = 8;
const int TestRepeats = 4;

int g_NumMasters = 0;
volatile intptr_t *g_LeavesExecuted = NULL;

int g_TestFailures[NumTests];
int g_CurConfig = 0;

int P = 0;

#if !__TBB_TASK_PRIORITY
namespace tbb {
    enum priority_t {
        priority_low = 0,
        priority_normal = 1,
        priority_high = 2
    };
}
#endif /* __TBB_TASK_PRIORITY */

tbb::priority_t Low = tbb::priority_normal,
                High = tbb::priority_high;
int PreemptionActivatorId = 1;

enum Options {
    NoPriorities = 0,
    TestPreemption = 1,
    Flog = 2,
    FlogEncloser = Flog | 4
};

const char *PriorityName(tbb::priority_t p) {
    if( p == tbb::priority_high ) return "high";
    if( p == tbb::priority_normal ) return "normal";
    if( p == tbb::priority_low ) return "low";
    return "bad";
}

void PrepareGlobals ( int numMasters ) {
    ASSERT( !g_NumMasters && !g_LeavesExecuted, NULL );
    g_NumMasters = numMasters;
    if ( !g_LeavesExecuted )
        g_LeavesExecuted = new intptr_t[numMasters];
    g_CurConfig = 0;
    memset( const_cast<intptr_t*>(g_LeavesExecuted), 0, sizeof(intptr_t) * numMasters );
    memset( g_TestFailures, 0, sizeof(int) * NumTests );
}

void ClearGlobals () {
    ASSERT( g_LeavesExecuted, NULL );
    delete [] g_LeavesExecuted;
    g_LeavesExecuted = NULL;
    g_NumMasters = 0;
    REMARK("\r                                                             \r");
}

class LeafTask : public tbb::task {
    int m_tid;
    uintptr_t m_opts;

    tbb::task* execute () {
        volatile int anchor = 0;
        for ( int i = 0; i < NumIterations; ++i )
            anchor += i;
        __TBB_FetchAndAddW(g_LeavesExecuted + m_tid, 1);
#if __TBB_TASK_PRIORITY
        ASSERT( !m_opts || (m_opts & Flog) || (!(m_opts & TestPreemption) ^ (m_tid == PreemptionActivatorId)), NULL );
        if ( (m_opts & TestPreemption) && g_LeavesExecuted[0] > P && group_priority() == tbb::priority_normal ) {
            ASSERT( m_tid == PreemptionActivatorId, NULL );
            ASSERT( (PreemptionActivatorId == 1 ? High > tbb::priority_normal : Low < tbb::priority_normal), NULL );
            set_group_priority( PreemptionActivatorId == 1 ? High : Low );
        }
#endif /* __TBB_TASK_PRIORITY */
        return NULL;
    }
public:
    LeafTask ( int tid, uintptr_t opts ) : m_tid(tid), m_opts(opts) {
        ASSERT( tid < g_NumMasters, NULL );
    }
};

template<class NodeType>
class NodeTask : public tbb::task {
protected:
    int m_tid;
    int m_depth;
    uintptr_t m_opts;
    task *m_root;

    void SpawnChildren ( task* parent_node ) {
        ASSERT( m_depth > 0, NULL );
        if ( g_LeavesExecuted[m_tid] % (100 / m_depth) == 0 ) {
            if ( m_opts & Flog ) {
#if __TBB_TASK_PRIORITY
                task *r = m_opts & FlogEncloser ? this : m_root;
                tbb::priority_t p = r->group_priority();
                r->set_group_priority( p == Low ? High : Low );
#endif /* __TBB_TASK_PRIORITY */
            }
            else
                __TBB_Yield();
        }
        parent_node->set_ref_count(NumLeafTasks + 1);
        --m_depth;
        for ( int i = 0; i < NumLeafTasks; ++i ) {
            task *t = m_depth ? (task*) new(parent_node->allocate_child()) NodeType(m_tid, m_depth, m_opts, m_root) 
                              : (task*) new(parent_node->allocate_child()) LeafTask(m_tid, m_opts);
            task::spawn(*t);
        }
    }

public:
    NodeTask ( int tid, int _depth, uintptr_t opts, task *r = NULL )
        : m_tid(tid), m_depth(_depth), m_opts(opts), m_root(r)
    {}
};

class NestedGroupNodeTask : public NodeTask<NestedGroupNodeTask> {
    task* execute () {
        tbb::task_group_context ctx; // Use bound context
        tbb::empty_task &r = *new( task::allocate_root(ctx) ) tbb::empty_task;
        SpawnChildren(&r);
        r.wait_for_all();
        task::destroy(r);
        return NULL;
    }
public:
    NestedGroupNodeTask ( int tid, int _depth, uintptr_t opts, task *r = NULL )
        : NodeTask<NestedGroupNodeTask>(tid, _depth, opts, r)
    {}
};

class BlockingNodeTask : public NodeTask<BlockingNodeTask> {
    task* execute () {
        SpawnChildren(this);
        wait_for_all();
        return NULL;
    }
public:
    BlockingNodeTask ( int tid, int _depth, uintptr_t opts, task *r = NULL )
        : NodeTask<BlockingNodeTask>(tid, _depth, opts, r) {}
};

class NonblockingNodeTask : public NodeTask<NonblockingNodeTask> {
    task* execute () {
        if ( m_depth < 0 )
            return NULL; // I'm just a continuation now
        recycle_as_safe_continuation();
        SpawnChildren(this);
        m_depth = -1;
        return NULL;
    }
public:
    NonblockingNodeTask ( int tid, int _depth, uintptr_t opts, task *r = NULL )
        : NodeTask<NonblockingNodeTask>(tid, _depth, opts, r)
    {}
};

template<class NodeType>
class MasterBodyBase : NoAssign, Harness::NoAfterlife {
protected:
    uintptr_t m_opts;

public:
    void RunTaskForest ( int id ) const {
        ASSERT( id < g_NumMasters, NULL );
        g_LeavesExecuted[id] = 0;
        int d = BaseDepth + id;
        tbb::task_scheduler_init init(P-1);
        tbb::task_group_context ctx (tbb::task_group_context::isolated);
        tbb::empty_task &r = *new( tbb::task::allocate_root(ctx) ) tbb::empty_task;
        const int R = 4;
        r.set_ref_count( R * P + 1 );
        // Only thread 1 changes its task tree priority in preemption test mode
        uintptr_t opts = m_opts & (id == PreemptionActivatorId ? ~0u : ~(uintptr_t)TestPreemption);
        for ( int i = 0; i < R; ++i ) {
            for ( int j = 1; j < P; ++j )
                r.spawn( *new(r.allocate_child()) NodeType(id, MinBaseDepth + id, opts, &r) );
            r.spawn( *new(r.allocate_child()) NodeType(id, d, opts, &r) );
        }
        int count = 1;
        intptr_t lastExecuted = 0;
        while ( r.ref_count() > 1 ) {
            // Give workers time to make some progress.
            for ( int i = 0; i < 10 * count; ++i )
                __TBB_Yield();
#if __TBB_TASK_PRIORITY
            if ( lastExecuted == g_LeavesExecuted[id] ) {
                // No progress. Likely all workers left to higher priority arena,
                // and then returned to RML. Request workers back from RML.
                tbb::task::enqueue( *new(tbb::task::allocate_root() ) tbb::empty_task, id == 0 ? Low : High );
                Harness::Sleep(count);
#if __TBB_ipf
                // Increased sleep periods are required on systems with unfair hyperthreading (Itanium(R) 2 processor)
                count += 10;
#endif
            }
            else {
                count = 1;
                lastExecuted = g_LeavesExecuted[id];
            }
#else /* !__TBB_TASK_PRIORITY */
            (void)lastExecuted;
            tbb::task::enqueue( *new(tbb::task::allocate_root() ) tbb::empty_task );
#endif /* !__TBB_TASK_PRIORITY */
        }
        ASSERT( g_LeavesExecuted[id] == R * ((1 << d) + ((P - 1) * (1 << (MinBaseDepth + id)))), NULL );
        g_LeavesExecuted[id] = -1;
        tbb::task::destroy(r);
    }

    MasterBodyBase ( uintptr_t opts ) : m_opts(opts) {}
};

template<class NodeType>
class MasterBody : public MasterBodyBase<NodeType> {
    int m_testIndex;
public:
    void operator() ( int id ) const {
        this->RunTaskForest(id);
        if ( this->m_opts & Flog )
            return;
        if ( this->m_opts & TestPreemption ) {
            if ( id == 1 && g_LeavesExecuted[0] == -1 ) {
                //REMARK( "Warning: Low priority master finished too early [depth %d]\n", Depth );
                ++g_TestFailures[m_testIndex];
            }
        }
        else {
            if ( id == 0 && g_LeavesExecuted[1] == -1 ) {
                //REMARK( "Warning: Faster master takes too long [depth %d]\n", Depth );
                ++g_TestFailures[m_testIndex];
            }
        }
    }

    MasterBody ( int idx, uintptr_t opts ) : MasterBodyBase<NodeType>(opts), m_testIndex(idx) {}
};

template<class NodeType>
void RunPrioritySwitchBetweenTwoMasters ( int idx, uintptr_t opts ) {
    ASSERT( idx < NumTests, NULL );
    REMARK( "Config %d: idx=%i, opts=%u\r", ++g_CurConfig, idx, (unsigned)opts );
    NativeParallelFor ( 2, MasterBody<NodeType>(idx, opts) );
    Harness::Sleep(50);
}

void TestPrioritySwitchBetweenTwoMasters () {
    REMARK( "Stress tests: %s / %s \n", Low == tbb::priority_low ? "Low" : "Normal", High == tbb::priority_normal ? "Normal" : "High" );
    PrepareGlobals( 2 );
    for ( int i = 0; i < TestRepeats; ++i ) {
        for ( BaseDepth = MinBaseDepth; BaseDepth <= MaxBaseDepth; ++BaseDepth ) {
            RunPrioritySwitchBetweenTwoMasters<BlockingNodeTask>( 0, NoPriorities );
            RunPrioritySwitchBetweenTwoMasters<BlockingNodeTask>( 1, TestPreemption );
            RunPrioritySwitchBetweenTwoMasters<NonblockingNodeTask>( 2, NoPriorities );
            RunPrioritySwitchBetweenTwoMasters<NonblockingNodeTask>( 3, TestPreemption );
            if ( i == 0 ) {
                RunPrioritySwitchBetweenTwoMasters<BlockingNodeTask>( 4, Flog );
                RunPrioritySwitchBetweenTwoMasters<NonblockingNodeTask>( 5, Flog );
                RunPrioritySwitchBetweenTwoMasters<NestedGroupNodeTask>( 6, Flog );
                RunPrioritySwitchBetweenTwoMasters<NestedGroupNodeTask>( 7, FlogEncloser );
            }
        }
    }
#if __TBB_TASK_PRIORITY
    const int NumRuns = TestRepeats * (MaxBaseDepth - MinBaseDepth + 1);
    for ( int i = 0; i < NumTests; ++i ) {
        if ( g_TestFailures[i] )
            REMARK( "Test %d: %d failures in %d runs\n", i, g_TestFailures[i], NumRuns );
        if ( g_TestFailures[i] * 100 / NumRuns > 50 ) {
            if ( i == 1 )
                REPORT( "Known issue: priority effect is limited in case of blocking-style nesting\n" );
            else
                REPORT( "Warning: test %d misbehaved too often (%d out of %d)\n", i, g_TestFailures[i], NumRuns );
        }
    }
#endif /* __TBB_TASK_PRIORITY */
    ClearGlobals();
}

class SingleChildRootTask : public tbb::task {
    tbb::task* execute () {
        set_ref_count(2);
        spawn ( *new(allocate_child()) tbb::empty_task );
        wait_for_all();
        return NULL;
    }
};

int TestSimplePriorityOps ( tbb::priority_t prio ) {
    tbb::task_scheduler_init init;
    tbb::task_group_context ctx;
#if __TBB_TASK_PRIORITY
    ctx.set_priority( prio );
#else /* !__TBB_TASK_PRIORITY */
    (void)prio;
#endif /* !__TBB_TASK_PRIORITY */
    tbb::task *r = new( tbb::task::allocate_root(ctx) ) tbb::empty_task;
    r->set_ref_count(2);
    r->spawn ( *new(r->allocate_child()) tbb::empty_task );
    REMARK( "TestSimplePriorityOps: waiting for a child\n" );
    r->wait_for_all();
    ASSERT( !r->ref_count(), NULL );
    REMARK( "TestLowPriority: executing an empty root\n" );
    tbb::task::spawn_root_and_wait(*r);
    r = new( tbb::task::allocate_root(ctx) ) SingleChildRootTask;
    REMARK( "TestLowPriority: executing a root with a single child\n" );
    tbb::task::spawn_root_and_wait(*r);
    return 0;
}

#include "tbb/parallel_for.h"

void EmulateWork( int ) {
    for ( int i = 0; i < 1000; ++i )
        __TBB_Yield();
}

class PeriodicActivitiesBody {
public:
    void operator() ( int id ) const {
        tbb::task_group_context ctx;
#if __TBB_TASK_PRIORITY
        ctx.set_priority( id ? High : Low );
#else /* !__TBB_TASK_PRIORITY */
        (void)id;
#endif /* !__TBB_TASK_PRIORITY */
        for ( int i = 0; i < 5; ++i ) {
            tbb::task_scheduler_init init;
            tbb::parallel_for( 1, 10000, &EmulateWork, ctx );
        }
    }
};

void TestPeriodicConcurrentActivities () {
    REMARK( "TestPeriodicConcurrentActivities: %s / %s \n", Low == tbb::priority_low ? "Low" : "Normal", High == tbb::priority_normal ? "Normal" : "High" );
    NativeParallelFor ( 2, PeriodicActivitiesBody() );
}

#include "harness_bad_expr.h"

void TestPriorityAssertions () {
#if TRY_BAD_EXPR_ENABLED && __TBB_TASK_PRIORITY
    REMARK( "TestPriorityAssertions\n" );
    tbb::priority_t bad_low_priority = tbb::priority_t( tbb::priority_low - 1 ),
                    bad_high_priority = tbb::priority_t( tbb::priority_high + 1 );
    tbb::task_group_context ctx;
    // Catch assertion failures
    tbb::set_assertion_handler( AssertionFailureHandler );
    TRY_BAD_EXPR( ctx.set_priority( bad_low_priority ), "Invalid priority level value" );
    tbb::task &t = *new( tbb::task::allocate_root() ) tbb::empty_task;
    TRY_BAD_EXPR( tbb::task::enqueue( t, bad_high_priority ), "Invalid priority level value" );
    // Restore normal assertion handling
    tbb::set_assertion_handler( NULL );
#endif /* TRY_BAD_EXPR_ENABLED && __TBB_TASK_PRIORITY */
}

#if __TBB_TASK_PRIORITY
tbb::atomic<tbb::priority_t> g_order;
tbb::atomic<bool> g_order_established;
class OrderedTask : public tbb::task {
    tbb::priority_t my_priority;
public:
    OrderedTask(tbb::priority_t p) : my_priority(p) {}
    tbb::task* execute() {
        tbb::priority_t prev = g_order.fetch_and_store(my_priority);
        if( my_priority != prev) {
            REMARK("prev:%s --> new:%s\n", PriorityName(prev), PriorityName(my_priority));
            // TODO: improve the test for concurrent workers
            if(!g_order_established) {
                // initial transition path allowed low->[normal]->high
                if(my_priority == tbb::priority_high)
                    g_order_established = true;
                else ASSERT(my_priority == tbb::priority_normal && prev == tbb::priority_low, NULL);
            } else { //transition path allowed high->normal->low
                if(prev == tbb::priority_high) ASSERT( my_priority == tbb::priority_normal, "previous priority is high - bad order");
                else if(prev == tbb::priority_normal) ASSERT( my_priority == tbb::priority_low, "previous priority is normal - bad order");
                else ASSERT(!g_order_established, "transition from low priority but not during initialization");
            }
        }
        EmulateWork(0);
        return NULL;
    }
    static void start(int i) {
        tbb::priority_t p = i%3==0? tbb::priority_low : (i%3==1? tbb::priority_normal : tbb::priority_high );
        OrderedTask &t = *new(tbb::task::allocate_root()) OrderedTask(p);
        tbb::task::enqueue(t, p);
    }
};

//Look for discussion of the issue at http://software.intel.com/en-us/forums/showthread.php?t=102159
void TestEnqueueOrder () {
    REMARK("Testing order of enqueued tasks\n");
    tbb::task_scheduler_init init(1); // to simplify transition checks only one extra worker for enqueue
    g_order = tbb::priority_low;
    g_order_established = false;
    for( int i = 0; i < 1000; i++)
        OrderedTask::start(i);
    while( g_order == tbb::priority_low ) __TBB_Yield();
    while( g_order != tbb::priority_low ) __TBB_Yield();
}
#endif /* __TBB_TASK_PRIORITY */

int TestMain () {
#if !__TBB_TASK_PRIORITY
    REMARK( "Priorities disabled: Running as just yet another task scheduler test\n" );
#else
    TestEnqueueOrder();
#endif /* __TBB_TASK_PRIORITY */
    TestPriorityAssertions();
    TestSimplePriorityOps(tbb::priority_low);
    TestSimplePriorityOps(tbb::priority_high);
    P = tbb::task_scheduler_init::default_num_threads();
    if ( P < 3 )
        return Harness::Skipped;
    TestPeriodicConcurrentActivities();
    TestPrioritySwitchBetweenTwoMasters();
    Low = tbb::priority_low;
    High = tbb::priority_normal;
    PreemptionActivatorId = 0;
    TestPeriodicConcurrentActivities();
    TestPrioritySwitchBetweenTwoMasters();
    High = tbb::priority_high;
    TestPeriodicConcurrentActivities();
    TestPrioritySwitchBetweenTwoMasters();
    PreemptionActivatorId = 1;
    TestPrioritySwitchBetweenTwoMasters();
    return Harness::Done;
}

#else /* !__TBB_TASK_GROUP_CONTEXT */

int TestMain () {
    return Harness::Skipped;
}

#endif /* !__TBB_TASK_GROUP_CONTEXT */
