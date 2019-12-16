; RUN: %opt < %s %loadEnzyme -enzyme -enzyme_preopt=false -mem2reg -instsimplify -simplifycfg -S -early-cse -instcombine -instsimplify | FileCheck %s

; Function Attrs: norecurse nounwind readnone uwtable
define dso_local double @square(double %x) #0 {
entry:
  %mul = fmul fast double %x, %x
  ret double %mul
}

; Function Attrs: noinline nounwind uwtable
define dso_local double @indirect(double (double)* nocapture %callee, double %x) local_unnamed_addr #1 {
entry:
  %call = tail call fast double %callee(double %x) #2
  ret double %call
}

; Function Attrs: noinline nounwind uwtable
define dso_local double @create(double %x) #1 {
entry:
  %call = tail call fast double @indirect(double (double)* nonnull @square, double %x)
  ret double %call
}

; Function Attrs: noinline nounwind uwtable
define dso_local double @derivative(double %x) local_unnamed_addr #1 {
entry:
  %0 = tail call double (double (double)*, ...) @__enzyme_autodiff(double (double)* nonnull @create, double %x)
  ret double %0
}

; Function Attrs: nounwind
declare double @__enzyme_autodiff(double (double)*, ...) #2

attributes #0 = { norecurse nounwind readnone uwtable "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="true" "no-jump-tables"="false" "no-nans-fp-math"="true" "no-signed-zeros-fp-math"="true" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="true" "use-soft-float"="false" }
attributes #1 = { noinline nounwind uwtable "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="true" "no-jump-tables"="false" "no-nans-fp-math"="true" "no-signed-zeros-fp-math"="true" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="true" "use-soft-float"="false" }
attributes #2 = { nounwind }

!llvm.module.flags = !{!0}
!llvm.ident = !{!1}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{!"clang version 7.1.0 "}

; CHECK: @"_enzyme_square'" = internal constant { { i8*, double } (double)*, { double } (double, double, i8*)* } { { i8*, double } (double)* @augmented_square, { double } (double, double, i8*)* @diffesquare }

; CHECK: define internal { double } @diffecreate(double %x, double %differeturn) #1 {
; CHECK-NEXT: entry:
; CHECK-NEXT:   %0 = call { double } @diffeindirect(double (double)* nonnull @square, double (double)* bitcast ({ { i8*, double } (double)*, { double } (double, double, i8*)* }* @"_enzyme_square'" to double (double)*), double %x, double %differeturn)
; CHECK-NEXT:   ret { double } %0
; CHECK-NEXT: }

; CHECK: define internal { i8*, double } @augmented_square(double %x)
; CHECK-NEXT: entry:
; CHECK-NEXT:   %malloccall = tail call noalias nonnull i8* @malloc(i64 0)
; CHECK-NEXT:   %mul = fmul fast double %x, %x
; CHECK-NEXT:   %0 = insertvalue { i8*, double } undef, i8* %malloccall, 0
; CHECK-NEXT:   %1 = insertvalue { i8*, double } %0, double %mul, 1
; CHECK-NEXT:   ret { i8*, double } %1
; CHECK-NEXT: }

; CHECK: define internal { double } @diffesquare(double %x, double %differeturn, i8* %tapeArg)
; CHECK-NEXT: entry:
; CHECK-NEXT:   tail call void @free(i8* nonnull %tapeArg)
; CHECK-NEXT:   %0 = fadd fast double %x, %x
; CHECK-NEXT:   %1 = fmul fast double %0, %differeturn
; CHECK-NEXT:   %2 = insertvalue { double } undef, double %1, 0
; CHECK-NEXT:   ret { double } %2
; CHECK-NEXT: }

; CHECK: define internal { double } @diffeindirect(double (double)* nocapture %callee, double (double)* %"callee'", double %x, double %differeturn)
; CHECK-NEXT: entry:
; CHECK-NEXT:   %0 = bitcast double (double)* %"callee'" to { i8*, double } (double)**
; CHECK-NEXT:   %1 = load { i8*, double } (double)*, { i8*, double } (double)** %0, align 8
; CHECK-NEXT:   %call_augmented = call { i8*, double } %1(double %x) #2
; CHECK-NEXT:   %2 = extractvalue { i8*, double } %call_augmented, 0
; CHECK-NEXT:   %3 = bitcast double (double)* %"callee'" to { double } (double, double, i8*)**
; CHECK-NEXT:   %4 = getelementptr { double } (double, double, i8*)*, { double } (double, double, i8*)** %3, i64 1
; CHECK-NEXT:   %5 = load { double } (double, double, i8*)*, { double } (double, double, i8*)** %4, align 8
; CHECK-NEXT:   %6 = call { double } %5(double %x, double %differeturn, i8* %2) #2
; CHECK-NEXT:   ret { double } %6
; CHECK-NEXT: }
