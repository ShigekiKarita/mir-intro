/*
  based on https://github.com/libmir/mir-glas/blob/master/bench/gemm_report.d
 */


import std.stdio;
import std.numeric : dotProduct;
import std.traits;
import std.datetime;
import std.conv;

import std.getopt;
import mir.random;
import mir.random.variable: UniformVariable;
import mir.random.algorithm: field;
import mir.ndslice;
import mir.utility: min;
import mir.internal.utility: isComplex, realType;
import mir.random;
import mir.random.algorithm;
import mir.random.variable;
import mir.ndslice;
import mir.internal.utility : fastmath;
import mir.math.sum : sum;


enum isVec(S) = packsOf!S == [1LU];

enum isMat(S) = packsOf!S == [2LU];


@fastmath
auto dotFor(S1, S2)(S1 s1, S2 s2) if(isVec!S1 && isVec!S2) {
    assert(s1.shape == s2.shape);
    double result = 0;
    foreach (i; 0 .. s1.length) {
        result += s1[i] * s2[i];
    }
    return result;
}

@fastmath
auto dotHF(S1, S2)(S1 s1, S2 s2) if(isVec!S1 && isVec!S2) {
    assert(s1.shape == s2.shape);
    return zip(s1, s2).map!"a * b".sum!"fast";
}


unittest {
   auto a = [1, 2, 3].sliced!double.universal;
   auto b = [2, 3, 4].sliced!double.universal;
   assert(dotFor(a, b) == dotProduct(a, b));
   assert(dotHF(a, b) == dotProduct(a, b));
}

// GEMM Pseudo_code: `C := alpha A × B + beta C`.
void gemm(alias dotFun, C)(C alpha,
                           Slice!(Universal, [2], const(C)*) asl,
                           Slice!(Universal, [2], const(C)*) bsl,
                           C beta,
                           Slice!(Universal, [2], C*) csl) {
    foreach (i; 0 .. csl.shape[0]) {
        foreach (j; 0 .. csl.shape[1]) {
            csl[i, j] = alpha * dotFun(asl[i], bsl[0 .. $, j]) + beta * csl[i, j];
        }
    }
}



alias C = float;
//alias C = double;
//alias C = cfloat;
//alias C = cdouble;

alias R = realType!C;

size_t[] reportValues = [
	10, 20, 30, 40, 50, 60, 70, 80, 90, 100,
	200, 300, 500, 600, 700, 800, 900, 1000,
	1200, 1400, 1600, 1800, 2000]; // .sliced.map!"a ^^ 2".ndarray;


void main(string[] args)
{
	size_t count = 100;
	auto helpInformation = 
        getopt(args,
               "count|c", "Iteration count. Default value is " ~ count.to!string, &count);
	if (helpInformation.helpWanted)
    {
            defaultGetoptPrinter("Parameters:", helpInformation.options);
            return;
    }
	auto rng = Random(unpredictableSeed);
	auto var = UniformVariable!int(-100, 100);
	writeln("m=n=k,FOR,HF,STD,MKL");


    static if(isComplex!C) {
        C alpha = 3 + 7i;
        C beta = 0 + 0i;
    }
    else {
        C alpha = 3;
        C beta = 0;
    }

	foreach(m; reportValues)
	{
		auto n = m;
		auto k = m;

		auto d = rng.field!C(var).slicedField(m, n).slice;
		auto c = rng.field!C(var).slicedField(m, n).slice;
		auto a = rng.field!C(var).slicedField(m, k).slice;
		auto b = rng.field!C(var).slicedField(k, n).slice;

		d[] = c[];
        import std.typecons;

		/// for-loop dot
		auto nsecsFor = double.max;
		foreach(_; 0..count)
		{
			StopWatch sw;
			sw.start;
            gemm!dotFor(alpha, a.universal, b.universal, beta, c.universal);
			sw.stop;
			auto newns = sw.peek.to!Duration.total!"nsecs".to!double;
			nsecsFor = min(newns, nsecsFor);
		}

        /// higher order function dot
		auto nsecsHF = double.max;
		foreach(_; 0..count)
		{
			StopWatch sw;
			sw.start;
            gemm!dotHF(alpha, a.universal, b.universal, beta, c.universal);
			sw.stop;
			auto newns = sw.peek.to!Duration.total!"nsecs".to!double;
			nsecsHF = min(newns, nsecsHF);
		}

        /// std.numeric.dotProduct
		auto nsecsStd = double.max;
		foreach(_; 0..count)
		{
			StopWatch sw;
			sw.start;
            gemm!dotProduct(alpha, a.universal, b.universal, beta, c.universal);
			sw.stop;
			auto newns = sw.peek.to!Duration.total!"nsecs".to!double;
			nsecsStd = min(newns, nsecsStd);
		}

        /// lubeck.mtimes (MKL)
        import lubeck;
		auto nsecsMKL = double.max;
		foreach(_; 0..count)
		{
			StopWatch sw;
			sw.start;
            c[] = mtimes(a.universal, b.universal);
			sw.stop;
			auto newns = sw.peek.to!Duration.total!"nsecs".to!double;
			nsecsMKL = min(newns, nsecsMKL);
		}

		/// Result
        writefln("%s,%s,%s,%s,%s", m,
                 nsecsFor,
                 nsecsHF,
                 nsecsStd,
                 nsecsMKL
            );
	}
}

