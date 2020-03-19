; RUN: %opt < %s %loadEnzyme -enzyme -enzyme_preopt=false -inline -mem2reg -instsimplify -adce -loop-deletion -correlated-propagation -simplifycfg -S | FileCheck %s

define void @sum(i64* %x, i64 %n) {
entry:
  %cmp = icmp eq i64 %n, 0
  br i1 %cmp, label %one, label %two

one:
  %phi1 = phi i64 [ 0, %entry ], [ %phi2, %two ]
  %cmp1 = icmp eq i64 %n, 1
  br i1 %cmp1, label %end, label %two

two:
  %phi2 = phi i64 [ 12, %entry ], [ %phi1, %one ]
  %cmp2 = icmp eq i64 %n, 2
  br i1 %cmp2, label %end, label %one

end:
  %phi3 = phi i64 [ %phi1, %one ], [ %phi2, %two ]
  store i64 %phi3, i64* %x
  ret void
}

; Function Attrs: nounwind uwtable
define dso_local void @dsum(i64* %x, i64* %xp, i64 %n) local_unnamed_addr #1 {
entry:
  call void (void (i64*, i64)*, ...) @__enzyme_autodiff(void (i64*, i64)* nonnull @sum, metadata !"diffe_dup", i64* %x, i64* %xp, i64 %n)
  ret void
}

; Function Attrs: nounwind
declare void @__enzyme_autodiff(void (i64*, i64)*, ...) #2

attributes #0 = { norecurse nounwind readonly uwtable }
attributes #1 = { nounwind uwtable } 
attributes #2 = { nounwind }

; CHECK: define dso_local void @dsum(i64* %x, i64* %xp, i64 %n) local_unnamed_addr #0 {
; CHECK-NEXT: entry:
; CHECK-NEXT:   %cmp.i = icmp eq i64 %n, 0
; CHECK-NEXT:   br i1 %cmp.i, label %one.i, label %two.i

; CHECK: one.i:                                            ; preds = %two.i, %entry
; CHECK-NEXT:   %_cache.0.i = phi i1 [ false, %entry ], [ true, %two.i ]
; CHECK-NEXT:   %phi1.i = phi i64 [ 0, %entry ], [ %phi2.i, %two.i ]
; CHECK-NEXT:   %cmp1.i = icmp eq i64 %n, 1
; CHECK-NEXT:   br i1 %cmp1.i, label %end.i, label %two.i

; CHECK: two.i:                                            ; preds = %one.i, %entry
; CHECK-NEXT:   %_cache1.0.i = phi i1 [ true, %one.i ], [ false, %entry ]
; CHECK-NEXT:   %phi2.i = phi i64 [ 12, %entry ], [ %phi1.i, %one.i ]
; CHECK-NEXT:   %cmp2.i = icmp eq i64 %n, 2
; CHECK-NEXT:   br i1 %cmp2.i, label %end.i, label %one.i

; CHECK: end.i:                                            ; preds = %two.i, %one.i
; CHECK-NEXT:   %_cache2.0.i = phi i1 [ false, %one.i ], [ true, %two.i ]
; CHECK-NEXT:   %_cache1.1.i = phi i1 [ true, %one.i ], [ %_cache1.0.i, %two.i ]
; CHECK-NEXT:   %_cache.1.i = phi i1 [ %_cache.0.i, %one.i ], [ true, %two.i ]
; CHECK-NEXT:   %phi3.i = phi i64 [ %phi1.i, %one.i ], [ %phi2.i, %two.i ]
; CHECK-NEXT:   store i64 %phi3.i, i64* %xp
; CHECK-NEXT:   store i64 %phi3.i, i64* %x
; CHECK-NEXT:   %brmerge.i = or i1 %_cache2.0.i, %_cache.1.i
; CHECK-NEXT:   br i1 %brmerge.i, label %inverttwo.i, label %diffesum.exit

; CHECK: inverttwo.i:                                      ; preds = %end.i, %inverttwo.i
; CHECK-NEXT:   %_cache1.1.not.i = xor i1 %_cache1.1.i, true
; CHECK-NEXT:   %_cache.1.not.i = xor i1 %_cache.1.i, true
; CHECK-NEXT:   %brmerge3.i = or i1 %_cache1.1.not.i, %_cache.1.not.i
; CHECK-NEXT:   br i1 %brmerge3.i, label %diffesum.exit, label %inverttwo.i

; CHECK: diffesum.exit:                                    ; preds = %inverttwo.i, %end.i
; CHECK-NEXT:   ret void
; CHECK-NEXT: }
