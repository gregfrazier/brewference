package com.epicmonstrosity.brewference.gguf;

import com.epicmonstrosity.brewference.generation.GenerationOptions;
import com.epicmonstrosity.brewference.gguf.data.Config;
import com.epicmonstrosity.brewference.gguf.loader.GgufCheckpointLoader;
import com.epicmonstrosity.brewference.tensor.QuantizedSegmentTensor;
import com.epicmonstrosity.brewference.tensor.QuantizedTensor;
import com.epicmonstrosity.brewference.model.ModelRunnerFactory;
import org.junit.jupiter.api.Test;

import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardCopyOption;

import static org.junit.jupiter.api.Assertions.*;

class GgufCheckpointLoaderLifecycleTest {
    @Test
    void mappedWeightsBecomeInaccessibleAfterClose() throws IOException {
        final Path file = writeMinimalGguf("test");
        final TestCheckpointLoader loader = new TestCheckpointLoader(file.toString());
        try {
            loader.loadWeights();
            final QuantizedTensor raw = loader.getWeights().tokenEmbeddingTable;
            assertInstanceOf(QuantizedSegmentTensor.class, raw);
            assertEquals(7, ((QuantizedSegmentTensor) raw).quantizedValue(0));
            assertEquals(raw, loader.getWeights().classifier);

            loader.close();

            assertThrows(IllegalStateException.class, () -> raw.value(0));
        } finally {
            loader.close();
            Files.deleteIfExists(file);
        }
    }

    @Test
    void mappedQ4WeightsLoadTieClassifierAndBecomeInaccessibleAfterClose() throws IOException {
        final Path file = writeMinimalGguf("test", 2);
        final TestCheckpointLoader loader = new TestCheckpointLoader(file.toString());
        try {
            loader.loadWeights();
            final QuantizedTensor weights = loader.getWeights().tokenEmbeddingTable;
            assertInstanceOf(QuantizedSegmentTensor.class, weights);
            assertEquals(-1, ((QuantizedSegmentTensor) weights).quantizedValue(0));
            assertEquals(weights, loader.getWeights().classifier);

            loader.close();

            assertThrows(IllegalStateException.class, () -> weights.value(0));
        } finally {
            loader.close();
            Files.deleteIfExists(file);
        }
    }

    @Test
    void truncatedQ4PayloadReportsItsTensorName() throws IOException {
        final Path file = writeMinimalGguf("test", 2, 17);
        final TestCheckpointLoader loader = new TestCheckpointLoader(file.toString());
        try {
            final IOException exception = assertThrows(IOException.class, loader::loadWeights);
            assertTrue(exception.getMessage().contains("token_embd.weight"));
        } finally {
            loader.close();
            Files.deleteIfExists(file);
        }
    }

    @Test
    void constructorFailureReleasesMappedFile() throws IOException {
        final Path file = Files.createTempFile("invalid", ".gguf");
        Files.write(file, new byte[]{'G', 'G', 'U', 'F'});

        assertThrows(IOException.class, () -> new TestCheckpointLoader(file.toString()));

        assertCanMove(file);
    }

    @Test
    void factoryClosesDiscoveryLoaderOnUnsupportedArchitecture() throws IOException {
        final Path file = writeMinimalGguf("unsupported");

        assertThrows(IllegalArgumentException.class,
                () -> ModelRunnerFactory.createModelRunner(file.toString(), new GenerationOptions(), new NoOpTokenConsumer()));

        assertCanMove(file);
    }

    private static Path writeMinimalGguf(final String architecture) throws IOException {
        return writeMinimalGguf(architecture, 8);
    }

    private static Path writeMinimalGguf(final String architecture, final int tensorType) throws IOException {
        return writeMinimalGguf(architecture, tensorType, tensorType == 2 ? 18 : 34);
    }

    private static Path writeMinimalGguf(final String architecture, final int tensorType, final int payloadBytes) throws IOException {
        final ByteArrayOutputStream output = new ByteArrayOutputStream();
        writeInt(output, 0x46554747);
        writeInt(output, 3);
        writeLong(output, 1);
        writeLong(output, 4);
        writeStringMetadata(output, "general.architecture", architecture);
        writeStringMetadata(output, "general.name", "test");
        writeStringMetadata(output, "tokenizer.ggml.model", "llama");
        writeStringMetadata(output, "tokenizer.ggml.pre", "default");
        writeString(output, "token_embd.weight");
        writeInt(output, 1);
        writeLong(output, 32);
        writeInt(output, tensorType);
        writeLong(output, 0);
        while (output.size() % 32 != 0) {
            output.write(0);
        }
        for (int i = 0; i < payloadBytes; i++) {
            if (i < 2) output.write(0);
            else output.write(tensorType == 2 ? 0x77 : 7);
        }

        final Path file = Files.createTempFile("minimal", ".gguf");
        Files.write(file, output.toByteArray());
        return file;
    }

    private static void assertCanMove(final Path file) throws IOException {
        final Path moved = file.resolveSibling(file.getFileName() + ".moved");
        Files.move(file, moved, StandardCopyOption.REPLACE_EXISTING);
        Files.deleteIfExists(moved);
    }

    private static void writeStringMetadata(final ByteArrayOutputStream output, final String key, final String value) {
        writeString(output, key);
        writeInt(output, 8);
        writeString(output, value);
    }

    private static void writeString(final ByteArrayOutputStream output, final String value) {
        final byte[] bytes = value.getBytes(java.nio.charset.StandardCharsets.UTF_8);
        writeLong(output, bytes.length);
        output.writeBytes(bytes);
    }

    private static void writeInt(final ByteArrayOutputStream output, final int value) {
        output.writeBytes(ByteBuffer.allocate(Integer.BYTES).order(ByteOrder.LITTLE_ENDIAN).putInt(value).array());
    }

    private static void writeLong(final ByteArrayOutputStream output, final long value) {
        output.writeBytes(ByteBuffer.allocate(Long.BYTES).order(ByteOrder.LITTLE_ENDIAN).putLong(value).array());
    }

    private static final class TestCheckpointLoader extends GgufCheckpointLoader {
        private TestCheckpointLoader(final String filename) throws IOException {
            super(filename, new GenerationOptions());
        }

        @Override
        protected Config parseConfig(final GenerationOptions options) {
            return new Config();
        }
    }

    private static final class NoOpTokenConsumer implements com.epicmonstrosity.brewference.generation.TokenConsumer {
        @Override
        public void onPrefillTotal(final int total) { }

        @Override
        public void onPrefillToken(final int position, final int tokenId, final String tokenText) { }

        @Override
        public void onGeneratedToken(final int position, final int tokenId, final String tokenText) { }

        @Override
        public void onComplete(final com.epicmonstrosity.brewference.generation.GenerationResult result) { }

        @Override
        public void onDebug(final String debug) { }
    }
}