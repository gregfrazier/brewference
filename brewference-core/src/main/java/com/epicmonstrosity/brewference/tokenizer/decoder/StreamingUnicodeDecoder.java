package com.epicmonstrosity.brewference.tokenizer.decoder;

import com.epicmonstrosity.brewference.generation.TokenConsumer;

import java.io.ByteArrayOutputStream;
import java.nio.charset.StandardCharsets;

/**
 * The StreamingUnicodeDecoder class processes and decodes incoming UTF-8 byte sequences
 * in a streaming manner, ensuring proper handling of incomplete multi-byte sequences
 * at the boundaries. It sends decoded strings to a provided {@link TokenConsumer}.
 * <p>
 * This class maintains a buffer for unprocessed byte sequences and manages
 * the conversion of these bytes into valid UTF-8 strings. Once decoding is
 * complete up to a certain point, it notifies the {@link TokenConsumer} with
 * the decoded string along with its associated metadata such as position and token ID.
 * <p>
 * The class ensures tracking and partial processing of byte streams,
 * making it well-suited for scenarios involving continuous data streams,
 * such as token generation or streaming APIs.
 * <p>
 * Constructor Summary:
 * - {@link StreamingUnicodeDecoder#StreamingUnicodeDecoder(TokenConsumer)}
 *
 * Method Summary:
 * - {@link StreamingUnicodeDecoder#append(byte[])}: Appends a byte sequence to the buffer and decodes it if possible.
 * - {@link StreamingUnicodeDecoder#append(int, int, byte[])}: Appends a byte sequence to the buffer, associates it with metadata, and decodes it if possible.
 * - {@link StreamingUnicodeDecoder#flush(int, int)}: Flushes the remaining buffer content to the consumer and clears the buffer.
 */
public class StreamingUnicodeDecoder {
    private final ByteArrayOutputStream pending = new ByteArrayOutputStream();
    private final TokenConsumer consumer;

    public StreamingUnicodeDecoder(final TokenConsumer consumer) {
        this.consumer = consumer;
    }

    public void append(final byte[] tokenBytes) {
        append(0, 0, tokenBytes);
    }

    public void append(final int position, final int tokenId, final byte[] tokenBytes) {
        pending.write(tokenBytes, 0, tokenBytes.length);
        final byte[] buf = pending.toByteArray();
        final int complete = lengthOfCompleteUtf8(buf);
        if (complete > 0) {
            consumer.onGeneratedToken(position, tokenId, new String(buf, 0, complete, StandardCharsets.UTF_8));
            pending.reset();
            pending.write(buf, complete, buf.length - complete);
        }
    }

    public void flush(final int position, final int tokenId) {
        final byte[] buf = pending.toByteArray();
        if (buf.length > 0) {
            consumer.onGeneratedToken(position, tokenId, new String(buf, StandardCharsets.UTF_8));
            pending.reset();
        }
    }

    private static int lengthOfCompleteUtf8(final byte[] buf) {
        // Walk back from the end past any trailing incomplete sequence
        int i = buf.length;
        int back = 0;
        while (i > 0 && back < 4) {
            final int b = buf[--i] & 0xFF;
            back++;
            if ((b & 0xC0) != 0x80) {
                final int need = determineUtf8CharacterSize(b);
                return (back >= need) ? buf.length : i;
            }
        }
        return buf.length;
    }

    private static int determineUtf8CharacterSize(final int byteValue) {
        if ((byteValue & 0x80) == 0x00)
            return 1;
        if ((byteValue & 0xE0) == 0xC0)
            return 2;
        if ((byteValue & 0xF0) == 0xE0)
            return 3;
        if ((byteValue & 0xF8) == 0xF0)
            return 4;
        return 1;
    }
}