// RUN: onnx-mlir-opt --aisle-tile --canonicalize --cse --split-input-file %s | FileCheck %s

// Gemm [12,32] x [32,48] + C: 3 independent N-tiles, 2 chained K-tiles each.
// C's window is the Y preload of K-tile 0; the tiles are concatenated.
func.func @gemm_k2_n3(%a: tensor<12x32xf16>, %b: tensor<32x48xf16>, %c: tensor<12x48xf16>) -> tensor<12x48xf16> {
  %as = "aisle.qconstant"() <{name = "A_shape", shape = [1, 4], value = dense<[[1, 1, 12, 32]]> : tensor<1x4xi32>}> : () -> tensor<1x4xi32>
  %bs = "aisle.qconstant"() <{name = "B_shape", shape = [1, 4], value = dense<[[1, 1, 32, 48]]> : tensor<1x4xi32>}> : () -> tensor<1x4xi32>
  %y = "aisle.GEMM"(%a, %as, %b, %bs, %c) <{transA = 0 : si64, transB = 0 : si64}> {onnx_node_name = "proj"} : (tensor<12x32xf16>, tensor<1x4xi32>, tensor<32x48xf16>, tensor<1x4xi32>, tensor<12x48xf16>) -> tensor<12x48xf16>
  return %y : tensor<12x48xf16>
}
// CHECK-LABEL: func.func @gemm_k2_n3
// CHECK-SAME:    ([[A:%.+]]: tensor<12x32xf16>, [[B:%.+]]: tensor<32x48xf16>, [[C:%.+]]: tensor<12x48xf16>)
// CHECK:       [[C0:%.+]] = "aisle.Window"([[C]]) <{offsets = array<i64: 0, 0>}> : (tensor<12x48xf16>) -> tensor<12x16xf16>
// CHECK:       [[A0:%.+]] = "aisle.Window"([[A]]) <{offsets = array<i64: 0, 0>}> : (tensor<12x32xf16>) -> tensor<12x16xf16>
// CHECK:       [[B00:%.+]] = "aisle.Window"([[B]]) <{offsets = array<i64: 0, 0>}> : (tensor<32x48xf16>) -> tensor<16x16xf16>
// CHECK:       [[Y00:%.+]] = "aisle.GEMM"([[A0]], {{%.+}}, [[B00]], {{%.+}}, [[C0]]) {{.*}}{onnx_node_name = "proj[n0,k0]"}
// CHECK:       [[A1:%.+]] = "aisle.Window"([[A]]) <{offsets = array<i64: 0, 16>}>
// CHECK:       [[B10:%.+]] = "aisle.Window"([[B]]) <{offsets = array<i64: 16, 0>}>
// CHECK:       [[Y01:%.+]] = "aisle.GEMM"([[A1]], {{%.+}}, [[B10]], {{%.+}}, [[Y00]]) {{.*}}{onnx_node_name = "proj[n0,k1]"}
// CHECK:       [[C1:%.+]] = "aisle.Window"([[C]]) <{offsets = array<i64: 0, 16>}>
// CHECK:       [[B01:%.+]] = "aisle.Window"([[B]]) <{offsets = array<i64: 0, 16>}>
// CHECK:       [[Y10:%.+]] = "aisle.GEMM"([[A0]], {{%.+}}, [[B01]], {{%.+}}, [[C1]]) {{.*}}{onnx_node_name = "proj[n1,k0]"}
// CHECK:       [[B11:%.+]] = "aisle.Window"([[B]]) <{offsets = array<i64: 16, 16>}>
// CHECK:       [[Y11:%.+]] = "aisle.GEMM"([[A1]], {{%.+}}, [[B11]], {{%.+}}, [[Y10]]) {{.*}}{onnx_node_name = "proj[n1,k1]"}
// CHECK:       [[Y21:%.+]] = "aisle.GEMM"({{.*}}{onnx_node_name = "proj[n2,k1]"}
// CHECK:       [[Y:%.+]] = "aisle.Concat"([[Y01]], [[Y11]], [[Y21]]) <{axis = 1 : i64}> : (tensor<12x16xf16>, tensor<12x16xf16>, tensor<12x16xf16>) -> tensor<12x48xf16>
// CHECK:       return [[Y]]

// -----

// MatMul chain: the second product's K-windows select whole N-tiles of the
// first, so they fold to those tiles (no Window, no Concat left).
func.func @chain(%x: tensor<12x32xf16>, %w1: tensor<32x48xf16>, %w2: tensor<48x16xf16>) -> tensor<12x16xf16> {
  %s0 = "aisle.qconstant"() <{name = "A_shape", shape = [1, 4], value = dense<[[1, 1, 12, 32]]> : tensor<1x4xi32>}> : () -> tensor<1x4xi32>
  %s1 = "aisle.qconstant"() <{name = "B_shape", shape = [1, 4], value = dense<[[1, 1, 32, 48]]> : tensor<1x4xi32>}> : () -> tensor<1x4xi32>
  %u = "aisle.MatMul"(%x, %s0, %w1, %s1) : (tensor<12x32xf16>, tensor<1x4xi32>, tensor<32x48xf16>, tensor<1x4xi32>) -> tensor<12x48xf16>
  %s2 = "aisle.qconstant"() <{name = "A_shape", shape = [1, 4], value = dense<[[1, 1, 12, 48]]> : tensor<1x4xi32>}> : () -> tensor<1x4xi32>
  %s3 = "aisle.qconstant"() <{name = "B_shape", shape = [1, 4], value = dense<[[1, 1, 48, 16]]> : tensor<1x4xi32>}> : () -> tensor<1x4xi32>
  %y = "aisle.MatMul"(%u, %s2, %w2, %s3) : (tensor<12x48xf16>, tensor<1x4xi32>, tensor<48x16xf16>, tensor<1x4xi32>) -> tensor<12x16xf16>
  return %y : tensor<12x16xf16>
}
// CHECK-LABEL: func.func @chain
// CHECK:       [[U0:%.+]] = "aisle.GEMM"({{.*}}{onnx_node_name = "matmul[n0,k1]"}
// CHECK:       [[U1:%.+]] = "aisle.GEMM"({{.*}}{onnx_node_name = "matmul[n1,k1]"}
// CHECK:       [[U2:%.+]] = "aisle.GEMM"({{.*}}{onnx_node_name = "matmul[n2,k1]"}
// CHECK-NOT:   "aisle.Concat"
// CHECK:       [[Y0:%.+]] = "aisle.MatMul"([[U0]], {{.*}}{onnx_node_name = "matmul[n0,k0]"}
// CHECK:       [[Y1:%.+]] = "aisle.GEMM"([[U1]], {{.*}}, [[Y0]]) {{.*}}{onnx_node_name = "matmul[n0,k1]"}
// CHECK:       [[Y2:%.+]] = "aisle.GEMM"([[U2]], {{.*}}, [[Y1]]) {{.*}}{onnx_node_name = "matmul[n0,k2]"}
// CHECK:       return [[Y2]]

// -----

// Already one launch, f32, or not a multiple of 16: untouched.
func.func @untouched(%a: tensor<12x16xf16>, %b: tensor<16x16xf16>, %f: tensor<12x32xf32>, %g: tensor<32x16xf32>, %h: tensor<12x24xf16>, %i: tensor<24x16xf16>) -> (tensor<12x16xf16>, tensor<12x16xf32>, tensor<12x16xf16>) {
  %s = "aisle.qconstant"() <{name = "A_shape", shape = [1, 4], value = dense<[[1, 1, 12, 16]]> : tensor<1x4xi32>}> : () -> tensor<1x4xi32>
  %y0 = "aisle.MatMul"(%a, %s, %b, %s) : (tensor<12x16xf16>, tensor<1x4xi32>, tensor<16x16xf16>, tensor<1x4xi32>) -> tensor<12x16xf16>
  %y1 = "aisle.MatMul"(%f, %s, %g, %s) : (tensor<12x32xf32>, tensor<1x4xi32>, tensor<32x16xf32>, tensor<1x4xi32>) -> tensor<12x16xf32>
  %y2 = "aisle.MatMul"(%h, %s, %i, %s) : (tensor<12x24xf16>, tensor<1x4xi32>, tensor<24x16xf16>, tensor<1x4xi32>) -> tensor<12x16xf16>
  return %y0, %y1, %y2 : tensor<12x16xf16>, tensor<12x16xf32>, tensor<12x16xf16>
}
// CHECK-LABEL: func.func @untouched
// CHECK-NOT:   aisle.Window
// CHECK-COUNT-3: "aisle.MatMul"
// CHECK-NOT:   aisle.Window

// -----

// A constant B read only by tiled products is split at compile time: one
// onnx.Constant per tile (for transB already transposed, so the tiles carry
// transB = 0) and no Window of it; the original constant disappears.
func.func @split_constant(%a: tensor<12x32xf16>, %c: tensor<12x16xf16>) -> tensor<12x16xf16> {
  %bt = onnx.Constant dense<"0x00000018001C001E002000210022002300248024002580250026802600278027002840288028C028002940298029C029002A402A802AC02A002B402B802BC02B002C202C402C602C802CA02CC02CE02C002D202D402D602D802DA02DC02DE02D002E202E402E602E802EA02EC02EE02E002F202F402F602F802FA02FC02FE02F0030103020303030403050306030703080309030A030B030C030D030E030F0300031103120313031403150316031703180319031A031B031C031D031E031F0310032103220323032403250326032703280329032A032B032C032D032E032F0320033103320333033403350336033703380339033A033B033C033D033E033F03300340834103418342034283430343834403448345034583460346834703478348034883490349834A034A834B034B834C034C834D034D834E034E834F034F83400350835103518352035283530353835403548355035583560356835703578358035883590359835A035A835B035B835C035C835D035D835E035E835F035F83500360836103618362036283630363836403648365036583660366836703678368036883690369836A036A836B036B836C036C836D036D836E036E836F036F83600370837103718372037283730373837403748375037583760376837703778378037883790379837A037A837B037B837C037C837D037D837E037E837F037F8370038043808380C381038143818381C382038243828382C383038343838383C384038443848384C385038543858385C386038643868386C387038743878387C388038843888388C389038943898389C38A038A438A838AC38B038B438B838BC38C038C438C838CC38D038D438D838DC38E038E438E838EC38F038F438F838FC380039043908390C391039143918391C392039243928392C393039343938393C394039443948394C395039543958395C396039643968396C397039743978397C398039843988398C399039943998399C39A039A439A839AC39B039B439B839BC39C039C439C839CC39D039D439D839DC39E039E439E839EC39F039F439F839FC39003A043A083A0C3A103A143A183A1C3A203A243A283A2C3A303A343A383A3C3A403A443A483A4C3A503A543A583A5C3A603A643A683A6C3A703A743A783A7C3A803A843A883A8C3A903A943A983A9C3AA03AA43AA83AAC3AB03AB43AB83ABC3AC03AC43AC83ACC3AD03AD43AD83ADC3AE03AE43AE83AEC3AF03AF43AF83AFC3A003B043B083B0C3B103B143B183B1C3B203B243B283B2C3B303B343B383B3C3B403B443B483B4C3B503B543B583B5C3B603B643B683B6C3B703B743B783B7C3B803B843B883B8C3B903B943B983B9C3BA03BA43BA83BAC3BB03BB43BB83BBC3BC03BC43BC83BCC3BD03BD43BD83BDC3BE03BE43BE83BEC3BF03BF43BF83BFC3B"> : tensor<16x32xf16>
  %as = "aisle.qconstant"() <{name = "A_shape", shape = [1, 4], value = dense<[[1, 1, 12, 32]]> : tensor<1x4xi32>}> : () -> tensor<1x4xi32>
  %y = "aisle.GEMM"(%a, %as, %bt, %as, %c) <{transA = 0 : si64, transB = 1 : si64}> : (tensor<12x32xf16>, tensor<1x4xi32>, tensor<16x32xf16>, tensor<1x4xi32>, tensor<12x16xf16>) -> tensor<12x16xf16>
  return %y : tensor<12x16xf16>
}
// CHECK-LABEL: func.func @split_constant
// CHECK-NOT:   tensor<16x32xf16>
// CHECK-DAG:   [[W0:%.+]] = onnx.Constant dense<"0x0000002C0030003200340035003600370038803800398039003A803A003B803B0018202C1030103208340835083608370438843804398439043A843A043B843B001C402C2030203210341035103610370838883808398839083A883A083B883B001E602C3030303218341835183618370C388C380C398C390C3A8C3A0C3B8C3B0020802C4030403220342035203620371038903810399039103A903A103B903B0021A02C5030503228342835283628371438943814399439143A943A143B943B0022C02C6030603230343035303630371838983818399839183A983A183B983B0023E02C7030703238343835383638371C389C381C399C391C3A9C3A1C3B9C3B0024002D8030803240344035403640372038A0382039A039203AA03A203BA03B8024202D9030903248344835483648372438A4382439A439243AA43A243BA43B0025402DA030A03250345035503650372838A8382839A839283AA83A283BA83B8025602DB030B03258345835583658372C38AC382C39AC392C3AAC3A2C3BAC3B0026802DC030C03260346035603660373038B0383039B039303AB03A303BB03B8026A02DD030D03268346835683668373438B4383439B439343AB43A343BB43B0027C02DE030E03270347035703670373838B8383839B839383AB83A383BB83B8027E02DF030F03278347835783678373C38BC383C39BC393C3ABC3A3C3BBC3B"> : tensor<16x16xf16>
// CHECK-DAG:   [[W1:%.+]] = onnx.Constant dense<"0x0028002E0031003380348035803680374038C0384039C039403AC03A403BC03B4028202E1031103388348835883688374438C4384439C439443AC43A443BC43B8028402E2031203390349035903690374838C8384839C839483AC83A483BC83BC028602E3031303398349835983698374C38CC384C39CC394C3ACC3A4C3BCC3B0029802E40314033A034A035A036A0375038D0385039D039503AD03A503BD03B4029A02E50315033A834A835A836A8375438D4385439D439543AD43A543BD43B8029C02E60316033B034B035B036B0375838D8385839D839583AD83A583BD83BC029E02E70317033B834B835B836B8375C38DC385C39DC395C3ADC3A5C3BDC3B002A002F80318033C034C035C036C0376038E0386039E039603AE03A603BE03B402A202F90319033C834C835C836C8376438E4386439E439643AE43A643BE43B802A402FA031A033D034D035D036D0376838E8386839E839683AE83A683BE83BC02A602FB031B033D834D835D836D8376C38EC386C39EC396C3AEC3A6C3BEC3B002B802FC031C033E034E035E036E0377038F0387039F039703AF03A703BF03B402BA02FD031D033E834E835E836E8377438F4387439F439743AF43A743BF43B802BC02FE031E033F034F035F036F0377838F8387839F839783AF83A783BF83BC02BE02FF031F033F834F835F836F8377C38FC387C39FC397C3AFC3A7C3BFC3B"> : tensor<16x16xf16>
// CHECK:       "aisle.GEMM"({{%.+}}, {{%.+}}, [[W0]], {{%.+}}, {{%.+}}) <{transA = 0 : si64, transB = 0 : si64}>
// CHECK:       "aisle.GEMM"({{%.+}}, {{%.+}}, [[W1]], {{%.+}}, {{%.+}}) <{transA = 0 : si64, transB = 0 : si64}>
// CHECK-NOT:   "aisle.Window"(%{{.*}}) {{.*}} -> tensor<16x16xf16>

// -----

// Add(MatMul(A, B), C) with C [M, N] is one GEMM: C preloads Y of K-tile 0
// (firmware launch_bias), no Add launch.  C may drop leading 1s.  A MatMul
// with another user, or a broadcast C, is left alone.
func.func @fuse_add(%x: tensor<1x12x32xf16>, %w: tensor<32x16xf16>, %p: tensor<12x16xf16>, %q: tensor<1x16xf16>) -> (tensor<1x12x16xf16>, tensor<1x12x16xf16>, tensor<1x12x16xf16>) {
  %s = "aisle.qconstant"() <{name = "A_shape", shape = [1, 4], value = dense<[[1, 1, 12, 32]]> : tensor<1x4xi32>}> : () -> tensor<1x4xi32>
  %e = "aisle.MatMul"(%x, %s, %w, %s) {onnx_node_name = "proj"} : (tensor<1x12x32xf16>, tensor<1x4xi32>, tensor<32x16xf16>, tensor<1x4xi32>) -> tensor<1x12x16xf16>
  %y = "aisle.Add"(%p, %s, %e, %s) : (tensor<12x16xf16>, tensor<1x4xi32>, tensor<1x12x16xf16>, tensor<1x4xi32>) -> tensor<1x12x16xf16>
  %f = "aisle.MatMul"(%x, %s, %w, %s) {onnx_node_name = "shared"} : (tensor<1x12x32xf16>, tensor<1x4xi32>, tensor<32x16xf16>, tensor<1x4xi32>) -> tensor<1x12x16xf16>
  %z = "aisle.Add"(%f, %s, %q, %s) : (tensor<1x12x16xf16>, tensor<1x4xi32>, tensor<1x16xf16>, tensor<1x4xi32>) -> tensor<1x12x16xf16>
  return %y, %z, %f : tensor<1x12x16xf16>, tensor<1x12x16xf16>, tensor<1x12x16xf16>
}
// CHECK-LABEL: func.func @fuse_add
// CHECK-SAME:    ([[X:%.+]]: tensor<1x12x32xf16>, [[W:%.+]]: tensor<32x16xf16>, [[P:%.+]]: tensor<12x16xf16>, [[Q:%.+]]: tensor<1x16xf16>)
// CHECK:       [[Y0:%.+]] = "aisle.GEMM"({{%.+}}, {{%.+}}, {{%.+}}, {{%.+}}, [[P]]) {{.*}}{onnx_node_name = "proj[n0,k0]"}
// CHECK:       [[Y1:%.+]] = "aisle.GEMM"({{%.+}}, {{%.+}}, {{%.+}}, {{%.+}}, [[Y0]]) {{.*}}{onnx_node_name = "proj[n0,k1]"}
// CHECK:       "aisle.MatMul"({{.*}}{onnx_node_name = "shared[n0,k0]"}
// CHECK:       [[F:%.+]] = "aisle.GEMM"({{.*}}{onnx_node_name = "shared[n0,k1]"}
// CHECK:       [[Z:%.+]] = "aisle.Add"([[F]], {{%.+}}, [[Q]],
// CHECK:       return [[Y1]], [[Z]], [[F]]

// -----

// Partial tiles: M = 5 rows (zero padded at run time) and N = 4 < 16 columns
// (one narrow N-tile): the windows keep M and the tile width is N.
func.func @partial(%a: tensor<5x32xf16>, %b: tensor<32x4xf16>, %c: tensor<5x4xf16>) -> tensor<5x4xf16> {
  %s = "aisle.qconstant"() <{name = "A_shape", shape = [1, 4], value = dense<[[1, 1, 5, 32]]> : tensor<1x4xi32>}> : () -> tensor<1x4xi32>
  %y = "aisle.GEMM"(%a, %s, %b, %s, %c) <{transA = 0 : si64, transB = 0 : si64}> : (tensor<5x32xf16>, tensor<1x4xi32>, tensor<32x4xf16>, tensor<1x4xi32>, tensor<5x4xf16>) -> tensor<5x4xf16>
  return %y : tensor<5x4xf16>
}
// CHECK-LABEL: func.func @partial
// CHECK-SAME:    ([[A:%.+]]: tensor<5x32xf16>, [[B:%.+]]: tensor<32x4xf16>, [[C:%.+]]: tensor<5x4xf16>)
// CHECK-DAG:   [[A0:%.+]] = "aisle.Window"([[A]]) <{offsets = array<i64: 0, 0>}> : (tensor<5x32xf16>) -> tensor<5x16xf16>
// CHECK-DAG:   [[B0:%.+]] = "aisle.Window"([[B]]) <{offsets = array<i64: 0, 0>}> : (tensor<32x4xf16>) -> tensor<16x4xf16>
// CHECK:       [[Y0:%.+]] = "aisle.GEMM"([[A0]], {{%.+}}, [[B0]], {{%.+}}, [[C]]) {{.*}} -> tensor<5x4xf16>
// CHECK-DAG:   [[A1:%.+]] = "aisle.Window"([[A]]) <{offsets = array<i64: 0, 16>}> : (tensor<5x32xf16>) -> tensor<5x16xf16>
// CHECK-DAG:   [[B1:%.+]] = "aisle.Window"([[B]]) <{offsets = array<i64: 16, 0>}> : (tensor<32x4xf16>) -> tensor<16x4xf16>
// CHECK:       [[Y1:%.+]] = "aisle.GEMM"([[A1]], {{%.+}}, [[B1]], {{%.+}}, [[Y0]]) {{.*}} -> tensor<5x4xf16>
// CHECK:       return [[Y1]]
